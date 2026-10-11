#!/usr/bin/env python3
"""Standalone NC-RTED queue controller; never receives test labels or metrics."""
import argparse, contextlib, hashlib, json, os, shutil, signal, sqlite3, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nc_rted.queue import ArtifactReadUncertain, HardLimit, JobQueue, QueueError
from nc_rted.recovery import CheckpointReadUncertain, RecoveryError
from nc_rted.storage_lock import allocation_lock, ensure_directory
from nc_rted.worker_runtime import acquire_supervisor_lock, attempt_state, gpu_lock, group_live, group_members, group_state, process_live, process_starttime, write_journal

class ConclusiveOutputFailure(QueueError):
    """The exited producer's publication is present but invalid."""

class PublicationUncertain(QueueError):
    """I/O or atomic-commit failure; retain the attempt lease for retry."""

def poll_seconds(payload):
    value = payload.get("poll_seconds", 5)
    return min(5, max(.05, float(value))) if isinstance(value, (int, float)) else 5

def _sync_capture_tree(root):
    """Sync captured files plus every directory entry that makes them reachable."""
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle: os.fsync(handle.fileno())
        elif path.is_dir():
            descriptor=os.open(path,os.O_DIRECTORY); os.fsync(descriptor); os.close(descriptor)
    parent=root
    while True:
        descriptor=os.open(parent,os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
        if parent == parent.parent: break
        parent=parent.parent


def _seal_capture_tree(root):
    """Make the verified closure read-only for the trusted local operator."""
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o444 if path.is_file() else 0o555)
    root.chmod(0o555)
    _sync_capture_tree(root)


def capture_allocation_bytes(copies, directory_count, launch_bytes, block):
    """Charge every capture destination and the exact serialized launch contract."""
    return (sum(((path.stat().st_size + block - 1) // block) * block for path in copies.values()) +
            directory_count * block + ((len(launch_bytes) + block - 1) // block) * block)


def capture_formal_inputs(payload, run_dir):
    """Copy validated launch inputs before Popen so later path replacement cannot alter code."""
    root=Path(run_dir)/"immutable-inputs"
    runtime=Path(payload["runtime_config"]); admission=Path(payload["formal_admission"])
    repo=Path(__file__).resolve().parents[1]
    def captured_command():
        command=list(payload["command"])
        command[1]=str(root/"scripts"/"nc_rted_train.py")
        command[command.index("--config")+1]=str(root/"runtime.json")
        command[command.index("--admission")+1]=str(root/"admission.json")
        command.extend(["--captured-source-map",str(root/"source-map.json")])
        return command
    def closure():
        try:
            raw=admission.read_bytes()
            if hashlib.sha256(raw).hexdigest() != payload["formal_admission_sha256"]: raise QueueError("formal admission changed before capture")
            expected=json.loads(raw).get("source_files", {})
        except (OSError, ValueError) as error: raise QueueError("formal admission is invalid for capture") from error
        if not isinstance(expected, dict): raise QueueError("formal admission lacks source closure")
        paths={}
        for original, digest in expected.items():
            if not isinstance(original, str) or not isinstance(digest, str) or len(digest) != 64:
                raise QueueError("formal source closure is invalid")
            try: relative=Path(original).resolve().relative_to(repo)
            except ValueError as error: raise QueueError("formal source closure escapes the executable repository") from error
            if relative.parts[0] not in {"src", "scripts"}:
                raise QueueError("formal source closure has an unsupported repository path")
            paths[relative]=Path(original)
        if not paths: raise QueueError("formal source closure has no repository files")
        return expected, paths
    def verify_capture(sealed=True):
        if not root.is_dir(): raise QueueError("immutable formal input path is not a directory")
        if (not (root/"runtime.json").is_file() or not (root/"admission.json").is_file() or
                hashlib.sha256((root/"runtime.json").read_bytes()).hexdigest() != payload["runtime_config_sha256"] or
                hashlib.sha256((root/"admission.json").read_bytes()).hexdigest() != payload["formal_admission_sha256"]):
            raise QueueError("captured formal configuration differs from the admitted closure")
        try: admission_document=json.loads((root/"admission.json").read_text())
        except (OSError, ValueError) as error: raise QueueError("captured formal admission is invalid") from error
        expected=admission_document.get("source_files", {})
        if not isinstance(expected, dict): raise QueueError("captured formal admission lacks source closure")
        allowed={Path("runtime.json"), Path("admission.json"), Path("source-map.json"), Path("launch-contract.json")}
        for original, digest in expected.items():
            if not isinstance(original, str) or not isinstance(digest, str): raise QueueError("captured formal source closure is invalid")
            original_path=Path(original)
            try: relative=original_path.resolve().relative_to(repo)
            except ValueError as error: raise QueueError("captured formal source closure escapes repository") from error
            captured=root/relative
            if not captured.is_file() or hashlib.sha256(captured.read_bytes()).hexdigest() != digest:
                raise QueueError("captured formal source differs from admitted closure")
            allowed.add(relative)
        try:
            source_map=json.loads((root/"source-map.json").read_text())
        except (OSError, ValueError) as error: raise QueueError("captured formal source map is invalid") from error
        try:
            runtime_document=json.loads((root/"runtime.json").read_text())
            inherited=runtime_document["inherited"]; stage2=runtime_document["stage2_cache"]
            runtime_files=json.loads((root/"inherited-source-manifest.json").read_text())["files"]
            runtime_expected={"inherited_external_root":"inherited","inherited_source_manifest":"inherited-source-manifest.json","stage2_module":"stage2-module.py"}
            if (not isinstance(runtime_files,dict) or source_map != {"schema":"nc_rted_captured_source_map/v1", "files":{name:str(Path(name).resolve().relative_to(repo)) for name in expected},"runtime":runtime_expected} or
                    hashlib.sha256((root/"inherited-source-manifest.json").read_bytes()).hexdigest() != inherited["source_manifest_sha256"] or
                    hashlib.sha256((root/"stage2-module.py").read_bytes()).hexdigest() != stage2["module_sha256"]): raise ValueError
            for relative, digest in runtime_files.items():
                candidate=root/"inherited"/relative
                if (not isinstance(relative,str) or not isinstance(digest,str) or Path(relative).is_absolute() or ".." in Path(relative).parts or
                        not candidate.is_file() or hashlib.sha256(candidate.read_bytes()).hexdigest() != digest): raise ValueError
                allowed.add(Path("inherited")/relative)
            allowed.update({Path("inherited-source-manifest.json"),Path("stage2-module.py")})
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise QueueError("captured formal runtime source closure is invalid") from error
        if source_map.get("files") != {name:str(Path(name).resolve().relative_to(repo)) for name in expected}:
            raise QueueError("captured formal source map differs from admitted closure")
        try:
            launch=json.loads((root/"launch-contract.json").read_text())
            if (launch.get("schema") != "nc_rted_captured_launch/v1" or launch.get("command") != captured_command() or
                    launch.get("environment") != payload["execution_environment"] or not isinstance(launch.get("cuda_visible_devices"),str) or not launch["cuda_visible_devices"]): raise ValueError
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise QueueError("captured formal launch contract is invalid") from error
        for captured in root.rglob("*"):
            relative=captured.relative_to(root)
            if "__pycache__" in relative.parts or captured.suffix in {".pyc", ".pyo"}:
                raise QueueError("captured formal closure contains bytecode")
            if captured.is_file() and relative not in allowed:
                raise QueueError("captured formal closure contains an unadmitted file")
            if sealed and captured.stat().st_mode & 0o222:
                raise QueueError("captured formal closure is writable")
        if sealed and root.stat().st_mode & 0o222:
            raise QueueError("captured formal closure root is writable")
    if root.exists():
        # A controller can crash after the durable capture but before Popen.
        # Reuse only the exact, independently verified closure for that attempt.
        verify_capture()
        try: _sync_capture_tree(root)
        except OSError as error: raise QueueError("cannot durably recover immutable formal inputs") from error
    else:
        try:
            expected, paths=closure()
            inputs=[runtime, admission, *paths.values()]
            if not runtime.is_file() or hashlib.sha256(runtime.read_bytes()).hexdigest() != payload["runtime_config_sha256"]:
                raise QueueError("runtime configuration changed before capture")
            try:
                runtime_document=json.loads(runtime.read_text()); inherited=runtime_document["inherited"]; stage2=runtime_document["stage2_cache"]
                manifest=Path(inherited["source_manifest"]); runtime_files=json.loads(manifest.read_text())["files"]; external=Path(inherited["external_root"]); stage_module=Path(stage2["module"])
                if (hashlib.sha256(manifest.read_bytes()).hexdigest() != inherited["source_manifest_sha256"] or not isinstance(runtime_files,dict) or
                        not stage_module.is_file() or hashlib.sha256(stage_module.read_bytes()).hexdigest() != stage2["module_sha256"]): raise ValueError
                inherited_inputs=[]
                for relative, digest in runtime_files.items():
                    source_file=external/relative
                    if (not isinstance(relative,str) or not isinstance(digest,str) or Path(relative).is_absolute() or ".." in Path(relative).parts or
                            not source_file.is_file() or hashlib.sha256(source_file.read_bytes()).hexdigest() != digest): raise ValueError
                    inherited_inputs.append(source_file)
                inputs.extend([manifest, stage_module, *inherited_inputs])
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise QueueError("runtime source closure changed before capture") from error
            reserve=int(payload.get("min_free_bytes", 0))
            if reserve <= 0: raise QueueError("formal capture lacks a disk reserve")
            with allocation_lock(root.parent):
                ensure_directory(root.parent, reserve)
                block=max(4096, os.statvfs(root.parent).f_frsize)
                # Every copied file and each newly created directory consumes
                # complete allocation blocks; reserve the exact bounded closure.
                directories={root}
                for relative in paths:
                    directories.update(root/parent for parent in relative.parents)
                required=sum(((path.stat().st_size + block - 1) // block) * block for path in inputs)
                required += len(directories) * block
                if shutil.disk_usage(root.parent).free < reserve + required:
                    raise QueueError("formal capture would violate the admitted disk reserve")
                root.mkdir()
                shutil.copy2(runtime,root/"runtime.json"); shutil.copy2(admission,root/"admission.json")
                for relative, original in paths.items():
                    destination=root/relative; destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(original,destination)
                runtime_document=json.loads(runtime.read_text()); inherited=runtime_document["inherited"]; stage2=runtime_document["stage2_cache"]
                manifest=Path(inherited["source_manifest"]); runtime_files=json.loads(manifest.read_text())["files"]; external=Path(inherited["external_root"])
                if hashlib.sha256(manifest.read_bytes()).hexdigest() != inherited["source_manifest_sha256"] or not isinstance(runtime_files,dict): raise QueueError("runtime inherited source manifest changed before capture")
                shutil.copy2(manifest,root/"inherited-source-manifest.json")
                for relative, digest in runtime_files.items():
                    source_file=external/relative; destination=root/"inherited"/relative
                    if (not isinstance(relative,str) or not isinstance(digest,str) or Path(relative).is_absolute() or ".." in Path(relative).parts or
                            not source_file.is_file() or hashlib.sha256(source_file.read_bytes()).hexdigest() != digest): raise QueueError("runtime inherited source differs from manifest")
                    destination.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source_file,destination)
                stage_module=Path(stage2["module"])
                if not stage_module.is_file() or hashlib.sha256(stage_module.read_bytes()).hexdigest() != stage2["module_sha256"]: raise QueueError("runtime resolver source changed before capture")
                shutil.copy2(stage_module,root/"stage2-module.py")
                (root/"source-map.json").write_text(json.dumps({"schema":"nc_rted_captured_source_map/v1", "files":{str(original):str(relative) for relative,original in paths.items()},"runtime":{"inherited_external_root":"inherited","inherited_source_manifest":"inherited-source-manifest.json","stage2_module":"stage2-module.py"}},sort_keys=True,separators=(",", ":")))
                try:
                    _, attestation=__import__("nc_rted.resource_attestation", fromlist=["_bound_file"])._bound_file(payload["resource_attestation"],payload["resource_attestation_sha256"],"resource attestation")
                    execution=attestation["execution"]
                    if execution.get("environment") != payload["execution_environment"]: raise ValueError
                    gpu_uuid=execution["gpu_uuid"]
                except (OSError, ValueError, KeyError, TypeError) as error:
                    raise QueueError("cannot bind captured formal launch contract") from error
                (root/"launch-contract.json").write_text(json.dumps({"schema":"nc_rted_captured_launch/v1","command":captured_command(),"environment":payload["execution_environment"],"cuda_visible_devices":gpu_uuid},sort_keys=True,separators=(",", ":")))
                verify_capture(sealed=False)
                _sync_capture_tree(root)
                _seal_capture_tree(root)
                verify_capture()
        except OSError as error:
            raise QueueError("cannot capture immutable formal launch inputs") from error
    launch=json.loads((root/"launch-contract.json").read_text())
    environment=dict(launch["environment"]); environment["PYTHONPATH"]=str(root/"src")
    return launch["command"], environment, launch["cuda_visible_devices"]


def capture_formal_bundle_inputs(payload, run_dir):
    """Snapshot all four admitted members before the bundle process starts."""
    root=Path(run_dir)/"immutable-inputs"; bundle=Path(payload["bundle_config"])
    ensure_formal_directory(payload, root.parent, "formal bundle capture parent")
    repo=Path(__file__).resolve().parents[1]
    def captured_command():
        command=list(payload["command"])
        command[1]=str(root/"scripts"/"nc_rted_interleaved_formal.py")
        command[command.index("--bundle")+1]=str(root/"bundle.json")
        command[command.index("--captured-root")+1]=str(root)
        return command
    def closure():
        try:
            document=json.loads(bundle.read_text())
            if hashlib.sha256(bundle.read_bytes()).hexdigest()!=payload["bundle_config_sha256"]: raise ValueError
            source_manifest=Path(document["source_manifest"]); source_map=Path(document["captured_source_map"]["path"])
            if (hashlib.sha256(source_manifest.read_bytes()).hexdigest()!=document["source_manifest_sha256"] or
                    hashlib.sha256(source_map.read_bytes()).hexdigest()!=document["captured_source_map"]["sha256"]): raise ValueError
            source_document=json.loads(source_manifest.read_text()); source_map_document=json.loads(source_map.read_text())
            files=source_document["files"]; mapped=source_map_document["files"]; members=document["members"]
            if (not isinstance(files,dict) or not isinstance(mapped,dict) or
                    not isinstance(members,dict) or set(members)!={"A","U","S","F"}): raise ValueError
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise QueueError("formal bundle inputs changed before capture") from error
        copies={Path("bundle.json"):bundle, Path("bundle-source-manifest.json"):source_manifest,
                Path("source-map.json"):source_map}
        def add(relative, source):
            if (not isinstance(relative,Path) or relative.is_absolute() or ".." in relative.parts or
                    not source.is_file()): raise QueueError("formal bundle capture path is invalid")
            previous=copies.get(relative)
            if previous is not None and previous.resolve() != source.resolve():
                raise QueueError("formal bundle capture has colliding source destinations")
            copies[relative]=source
        for relative,digest in files.items():
            source=repo/relative
            if (not isinstance(relative,str) or not isinstance(digest,str) or len(digest)!=64 or
                    not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=digest):
                raise QueueError("formal bundle source differs before capture")
            add(Path(relative), source)
        for original,relative in mapped.items():
            source=Path(original)
            if (not isinstance(original,str) or not isinstance(relative,str) or not source.is_file()):
                raise QueueError("formal bundle captured source map differs")
            add(Path(relative), source)
        runtimes=[]
        for group in ("A","U","S","F"):
            item=members[group]; runtime_path=Path(item.get("runtime", "")); admission_path=Path(item.get("admission", ""))
            if (set(item)!={"runtime","runtime_sha256","admission","admission_sha256"} or
                    not runtime_path.is_file() or not admission_path.is_file() or
                    hashlib.sha256(runtime_path.read_bytes()).hexdigest()!=item["runtime_sha256"] or
                    hashlib.sha256(admission_path.read_bytes()).hexdigest()!=item["admission_sha256"]):
                raise QueueError("formal bundle member changed before capture")
            try:
                admission=json.loads(admission_path.read_text()); expected=admission["source_files"]
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise QueueError("formal bundle admission is invalid before capture") from error
            if not isinstance(expected,dict) or expected != {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in mapped}:
                raise QueueError("formal bundle admission differs from captured source map")
            add(Path("members")/group/"runtime.json", runtime_path)
            add(Path("members")/group/"admission.json", admission_path)
            runtimes.append(json.loads(runtime_path.read_text()))
        try:
            inherited=runtimes[0]["inherited"]; stage2=runtimes[0]["stage2_cache"]; binding=source_map_document["runtime"]
            inherited_manifest=Path(inherited["source_manifest"]); external=Path(inherited["external_root"]); stage_module=Path(stage2["module"])
            inherited_files=json.loads(inherited_manifest.read_text())["files"]
            if (not isinstance(inherited_files,dict) or
                    hashlib.sha256(inherited_manifest.read_bytes()).hexdigest()!=inherited["source_manifest_sha256"] or
                    not stage_module.is_file() or hashlib.sha256(stage_module.read_bytes()).hexdigest()!=stage2["module_sha256"]): raise ValueError
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise QueueError("formal bundle runtime source closure changed before capture") from error
        for runtime in runtimes[1:]:
            if runtime.get("inherited") != inherited or runtime.get("stage2_cache") != stage2:
                raise QueueError("formal bundle members do not share one runtime source closure")
        add(Path(binding["inherited_source_manifest"]), inherited_manifest)
        add(Path(binding["stage2_module"]), stage_module)
        for relative,digest in inherited_files.items():
            source=external/relative
            if (not isinstance(relative,str) or not isinstance(digest,str) or len(digest)!=64 or
                    not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=digest):
                raise QueueError("formal bundle inherited source differs before capture")
            add(Path(binding["inherited_external_root"])/relative, source)
        return document, copies
    def verify_capture(sealed=True):
        if not root.is_dir(): raise QueueError("immutable formal bundle input path is not a directory")
        copied_bundle=root/"bundle.json"
        try:
            document=json.loads(copied_bundle.read_text())
            if hashlib.sha256(copied_bundle.read_bytes()).hexdigest()!=payload["bundle_config_sha256"]: raise ValueError
            source_manifest=root/"bundle-source-manifest.json"; source_map=root/"source-map.json"
            if (hashlib.sha256(source_manifest.read_bytes()).hexdigest()!=document["source_manifest_sha256"] or
                    hashlib.sha256(source_map.read_bytes()).hexdigest()!=document["captured_source_map"]["sha256"]): raise ValueError
            source_document=json.loads(source_manifest.read_text()); source_map_document=json.loads(source_map.read_text())
            files=source_document["files"]; mapped=source_map_document["files"]; members=document["members"]
            if not isinstance(files,dict) or not isinstance(mapped,dict) or set(members)!={"A","U","S","F"}: raise ValueError
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise QueueError("captured formal bundle closure is invalid") from error
        allowed={Path("bundle.json"),Path("bundle-source-manifest.json"),Path("source-map.json"),Path("launch-contract.json")}
        for relative,digest in files.items():
            candidate=root/relative
            if (not isinstance(relative,str) or not isinstance(digest,str) or Path(relative).is_absolute() or ".." in Path(relative).parts or
                    not candidate.is_file() or hashlib.sha256(candidate.read_bytes()).hexdigest()!=digest): raise QueueError("captured formal bundle source differs")
            allowed.add(Path(relative))
        runtimes=[]
        for group in ("A","U","S","F"):
            item=members[group]; member=root/"members"/group
            runtime=member/"runtime.json"; admission=member/"admission.json"
            if (not runtime.is_file() or not admission.is_file() or
                    hashlib.sha256(runtime.read_bytes()).hexdigest()!=item.get("runtime_sha256") or
                    hashlib.sha256(admission.read_bytes()).hexdigest()!=item.get("admission_sha256")): raise QueueError("captured formal bundle member differs")
            try: expected=json.loads(admission.read_text())["source_files"]
            except (OSError, ValueError, KeyError, TypeError) as error: raise QueueError("captured formal bundle admission is invalid") from error
            if expected != {name: hashlib.sha256((root/relative).read_bytes()).hexdigest() for name,relative in mapped.items()}:
                raise QueueError("captured formal bundle source map differs from admission")
            allowed.update({Path("members")/group/"runtime.json",Path("members")/group/"admission.json"})
            runtimes.append(json.loads(runtime.read_text()))
        try:
            inherited=runtimes[0]["inherited"]; stage2=runtimes[0]["stage2_cache"]; binding=source_map_document["runtime"]
            manifest=root/binding["inherited_source_manifest"]; runtime_files=json.loads(manifest.read_text())["files"]
            if (not isinstance(runtime_files,dict) or hashlib.sha256(manifest.read_bytes()).hexdigest()!=inherited["source_manifest_sha256"] or
                    hashlib.sha256((root/binding["stage2_module"]).read_bytes()).hexdigest()!=stage2["module_sha256"]): raise ValueError
            for runtime in runtimes[1:]:
                if runtime.get("inherited")!=inherited or runtime.get("stage2_cache")!=stage2: raise ValueError
            for relative,digest in runtime_files.items():
                candidate=root/binding["inherited_external_root"]/relative
                if (not isinstance(relative,str) or not isinstance(digest,str) or Path(relative).is_absolute() or ".." in Path(relative).parts or
                        not candidate.is_file() or hashlib.sha256(candidate.read_bytes()).hexdigest()!=digest): raise ValueError
                allowed.add(Path(binding["inherited_external_root"])/relative)
            allowed.update({Path(binding["inherited_source_manifest"]),Path(binding["stage2_module"])})
            launch=json.loads((root/"launch-contract.json").read_text())
            if (launch != {"schema":"nc_rted_captured_bundle_launch/v1","command":captured_command(),
                           "environment":payload["execution_environment"],"cuda_visible_devices":launch.get("cuda_visible_devices")} or
                    not isinstance(launch["cuda_visible_devices"],str) or not launch["cuda_visible_devices"]): raise ValueError
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise QueueError("captured formal bundle runtime or launch contract is invalid") from error
        for candidate in root.rglob("*"):
            relative=candidate.relative_to(root)
            if "__pycache__" in relative.parts or candidate.suffix in {".pyc",".pyo"} or (candidate.is_file() and relative not in allowed):
                raise QueueError("captured formal bundle closure contains an unadmitted file")
            if sealed and candidate.stat().st_mode & 0o222: raise QueueError("captured formal bundle closure is writable")
        if sealed and root.stat().st_mode & 0o222: raise QueueError("captured formal bundle root is writable")
    if root.exists():
        verify_capture()
        try: _sync_capture_tree(root)
        except OSError as error: raise QueueError("cannot durably recover immutable formal bundle inputs") from error
    else:
        try:
            _document, copies=closure()
        except OSError as error:
            raise QueueError("cannot inspect formal bundle inputs") from error
        reserve=int(payload.get("min_free_bytes", 0))
        if reserve <= 0: raise QueueError("formal bundle capture lacks a disk reserve")
        try:
            _, attestation=__import__("nc_rted.resource_attestation", fromlist=["_bound_file"])._bound_file(payload["resource_attestation"],payload["resource_attestation_sha256"],"formal bundle resource attestation")
            execution=attestation["execution"]
            if execution.get("environment") != payload["execution_environment"]: raise ValueError
            launch_bytes=json.dumps({"schema":"nc_rted_captured_bundle_launch/v1","command":captured_command(),
                                     "environment":payload["execution_environment"],"cuda_visible_devices":execution["gpu_uuid"]},
                                    sort_keys=True,separators=(",", ":")).encode()
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise QueueError("cannot bind captured formal bundle launch contract") from error
        with allocation_lock(root.parent):
            ensure_directory(root.parent, reserve)
            block=max(4096, os.statvfs(root.parent).f_frsize)
            directories={root}
            for relative in copies: directories.update((root/relative).parents)
            # Every destination consumes an allocation, even when two capture paths share a source.
            required=capture_allocation_bytes(copies, len(directories), launch_bytes, block)
            if shutil.disk_usage(root.parent).free < reserve + required:
                raise QueueError("formal bundle capture would violate the admitted disk reserve")
            root.mkdir()
            for relative,source in copies.items():
                target=root/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
            (root/"launch-contract.json").write_bytes(launch_bytes)
            verify_capture(sealed=False)
            _sync_capture_tree(root); _seal_capture_tree(root)
            verify_capture()
    launch=json.loads((root/"launch-contract.json").read_text())
    environment=dict(launch["environment"]); environment["PYTHONPATH"]=str(root/"src")
    return launch["command"],environment,launch["cuda_visible_devices"]

def controller_may_live(owner):
    """Default owners include host:pid; do not steal from a live controller."""
    try:
        parts=owner.split(":")
        if len(parts) not in {2,3} or (len(parts)==3 and parts[1] != "recover"): return True
        host, raw_pid = parts[0], parts[-1]; pid = int(raw_pid)
        if host != os.uname().nodename: return True
        os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError):
        return False

def terminate_group(pid, starttime, known_members=None, timeout=30):
    """Signal only identity-verified members; unknown group state is retained."""
    if attempt_state(pid, starttime, os.uname().nodename) not in {"live", "group_live"}:
        return False
    def signal_verified(signum):
        members=group_members(pid)
        if members is None or (known_members is not None and not set(members.items()) <= set(known_members.items())):
            return None
        if not members: return {}
        descriptors=[]
        try:
            for member, member_start in members.items():
                descriptor=os.pidfd_open(member)
                if process_starttime(member) != member_start:
                    return None
                descriptors.append(descriptor)
            for descriptor in descriptors: signal.pidfd_send_signal(descriptor, signum)
        except (AttributeError, OSError):
            return None
        finally:
            for descriptor in descriptors: os.close(descriptor)
        return members
    if signal_verified(signal.SIGTERM) is None: return False
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        members=group_members(pid)
        if members == {}: return True
        if members is None or (known_members is not None and not set(members.items()) <= set(known_members.items())): return False
        time.sleep(.2)
    if signal_verified(signal.SIGKILL) is None: return False
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        members=group_members(pid)
        if members == {}: return True
        if members is None or (known_members is not None and not set(members.items()) <= set(known_members.items())): return False
        time.sleep(.1)
    return False

def observe_progress(queue, job, path, owner=None):
    if not path or not Path(path).is_file(): return
    try:
        record=json.loads(Path(path).read_text())
        if not isinstance(record,dict): raise ValueError
        counter=record["counter"]
        if (not isinstance(counter,int) or counter < 0 or record.get("job_key") != job["job_key"] or
                record.get("lease_token") != job["lease_token"]): raise ValueError
        evidence=Path(record.get("committed_path", ""))
        root=attempt_directory(json.loads(job["payload"]),job).resolve()
        if not evidence.is_file() or root not in evidence.resolve().parents:
            raise ValueError
        if evidence.resolve() == Path(path).resolve(): raise ValueError
        committed=json.loads(evidence.read_text())
        if not isinstance(committed,dict): raise ValueError
        if (committed.get("schema") != "nc_rted_progress_commit_v1" or committed.get("job_key") != job["job_key"] or
                committed.get("lease_token") != job["lease_token"] or committed.get("counter") != counter):
            raise ValueError
        artifact=Path(committed.get("artifact_path", ""))
        digest=committed.get("artifact_sha256")
        if (not artifact.is_file() or artifact.resolve() == Path(path).resolve() or root not in artifact.resolve().parents or not isinstance(digest,str) or len(digest)!=64 or
                hashlib.sha256(artifact.read_bytes()).hexdigest() != digest): raise ValueError
        transaction=committed.get("transaction_type")
        artifact_document=json.loads(artifact.read_text())
        if not isinstance(artifact_document,dict): raise ValueError
        if transaction == "checkpoint":
            payload=artifact.parent / "state.pt"
            expected=json.loads(job["payload"]).get("run_identity")
            if (artifact_document.get("schema") != "nc_rted_checkpoint_v2" or artifact_document.get("completed_updates") != counter or
                    not isinstance(expected,dict) or artifact_document.get("identity") != expected or not payload.is_file() or payload.resolve() == Path(path).resolve() or artifact_document.get("payload_bytes") != payload.stat().st_size or artifact_document.get("payload_sha256") != hashlib.sha256(payload.read_bytes()).hexdigest()): raise ValueError
            from nc_rted.recovery import validate_checkpoint_payload
            validate_checkpoint_payload(payload, artifact_document)
        elif transaction == "formal_bundle":
            payload=json.loads(job["payload"])
            boundary=Path(artifact_document.get("bundle_checkpoint", ""))
            digest=artifact_document.get("bundle_checkpoint_sha256")
            identities=payload.get("member_identities")
            if (artifact_document.get("schema") != "nc_rted_formal_bundle_progress/v1" or
                    artifact_document.get("member_identities") != identities or
                    artifact_document.get("counter") != counter or not boundary.is_file() or
                    not isinstance(digest,str) or hashlib.sha256(boundary.read_bytes()).hexdigest() != digest): raise ValueError
            expected_boundary=(Path(payload["bundle_checkpoint_root"])/"commits"/
                               ("final.json" if counter == 1000 else f"update_{counter:06d}.json"))
            if boundary.is_symlink() or boundary.resolve() != expected_boundary.resolve(): raise ValueError
            boundary_document=json.loads(boundary.read_text())
            if (boundary_document.get("schema") != "nc_rted_bundle_checkpoint_v1" or
                    boundary_document.get("completed_updates") != counter or
                    boundary_document.get("members") != identities or
                    set(boundary_document.get("checkpoints", {})) != {"A", "U", "S", "F"}): raise ValueError
            roots=payload.get("checkpoint_roots")
            if not isinstance(roots,list) or len(roots)!=4: raise ValueError
            from nc_rted.recovery import validate_checkpoint_payload
            for group,checkpoint_root in zip(("A","U","S","F"),roots):
                record=boundary_document["checkpoints"][group]
                directory=record.get("directory") if isinstance(record,dict) else None
                manifest=Path(checkpoint_root)/str(directory)/"manifest.json"
                state=manifest.parent/"state.pt"
                if (directory not in ({"final"} if counter == 1000 else {f"update_{counter:06d}"}) or
                        manifest.is_symlink() or state.is_symlink() or not manifest.is_file() or not state.is_file() or
                        record.get("manifest_sha256") != hashlib.sha256(manifest.read_bytes()).hexdigest()): raise ValueError
                manifest_document=json.loads(manifest.read_text())
                if (manifest_document.get("identity") != identities[group] or
                        manifest_document.get("completed_updates") != counter or
                        manifest_document.get("final") != (counter == 1000)):
                    raise ValueError
                validate_checkpoint_payload(state, manifest_document)
        elif transaction == "media":
            ids=artifact_document.get("committed_ids")
            results=artifact_document.get("results")
            if (artifact.resolve() == Path(path).resolve() or artifact_document.get("schema") != "nc_rted_media_commit_v1" or artifact_document.get("job_key") != job["job_key"] or artifact_document.get("lease_token") != job["lease_token"] or artifact_document.get("input_hash") != job.get("input_hash") or not isinstance(ids,list) or not ids or len(ids) != counter or len(set(ids)) != len(ids) or not isinstance(results,list) or len(results) != counter): raise ValueError
            result_ids=[]
            for result in results:
                if not isinstance(result,dict) or not isinstance(result.get("id"),str) or not result["id"]: raise ValueError
                candidate=Path(result.get("path", "")); checksum=result.get("sha256")
                if (not candidate.is_file() or root not in candidate.resolve().parents or candidate.resolve() == Path(path).resolve() or not isinstance(checksum,str) or len(checksum)!=64 or hashlib.sha256(candidate.read_bytes()).hexdigest()!=checksum): raise ValueError
                result_ids.append(result["id"])
            if result_ids != ids or len(set(result_ids)) != len(result_ids): raise ValueError
        else: raise ValueError
        queue.record_progress(job["job_key"],job["lease_token"],counter,owner=owner)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError, RecoveryError):
        return

def artifact_records(payload):
    records=[]
    for expected in payload["expected_outputs"]:
        artifact=Path(expected["path"])
        if not artifact.is_file(): raise RuntimeError(f"missing declared output {artifact}")
        records.append({"path":str(artifact),"checksum":hashlib.sha256(artifact.read_bytes()).hexdigest()})
    JobQueue.validate_artifacts(payload["expected_outputs"], records)
    return records

def attempt_directory(payload, job):
    return Path(payload["run_dir"]) / job["job_key"] / f"attempt-{job['attempts']}-{job['lease_token']}"


def ensure_formal_directory(payload, path, label):
    """Create one formal destination through canonical, non-symlink components."""
    volume=Path(payload["data_volume"]).resolve()
    try: volume_device=volume.stat().st_dev
    except OSError as error: raise QueueError(f"{label} volume is unavailable") from error
    path=Path(path)
    try: relative=path.relative_to(volume)
    except ValueError as error: raise QueueError(f"{label} escapes approved volume") from error
    if any(component in {"", ".", ".."} for component in relative.parts):
        raise QueueError(f"{label} is not canonical")
    current=volume
    for component in relative.parts:
        current=current/component
        if current.exists():
            if current.is_symlink() or not current.is_dir(): raise QueueError(f"{label} contains a symlink or non-directory")
        else:
            current.mkdir()
            descriptor=os.open(current.parent,os.O_DIRECTORY); os.fsync(descriptor); os.close(descriptor)
        try:
            if current.stat().st_dev != volume_device: raise QueueError(f"{label} is on an unapproved filesystem")
        except OSError as error: raise QueueError(f"{label} cannot be inspected") from error
    return path


def ensure_attempt_directory(payload, job):
    """Create a formal attempt only through non-symlink components on its volume."""
    path=attempt_directory(payload,job)
    if job.get("kind") not in {"formal_train", "formal_bundle", "formal_bundle_segment"}:
        path.mkdir(parents=True,exist_ok=True); return path
    return ensure_formal_directory(payload,path,"formal attempt path")


def ensure_checkpoint_directory(payload):
    root=ensure_formal_directory(payload,Path(payload["checkpoint_root"]),"formal checkpoint root")
    final=root / "final"
    # CheckpointStore publishes this directory atomically and refuses to
    # overwrite it, so validate an existing final without creating an empty one.
    if final.exists(): ensure_formal_directory(payload,final,"formal checkpoint path")
    return root


def ensure_bundle_checkpoint_directories(payload):
    """Prepare the four independent roots and their common commit root."""
    roots=payload.get("checkpoint_roots"); common=payload.get("bundle_checkpoint_root")
    if not isinstance(roots,list) or len(roots)!=4 or not isinstance(common,str):
        raise QueueError("formal bundle checkpoint destinations are incomplete")
    canonical=[ensure_formal_directory(payload,Path(value),"formal bundle member checkpoint root") for value in roots]
    canonical.append(ensure_formal_directory(payload,Path(common),"formal bundle checkpoint root"))
    if len(set(canonical)) != len(canonical) or any(path in other.parents or other in path.parents for index,path in enumerate(canonical) for other in canonical[index+1:]):
        raise QueueError("formal bundle checkpoint destinations overlap")
    for root in canonical[:4]:
        final=root/"final"
        if final.exists(): ensure_formal_directory(payload,final,"formal bundle checkpoint path")
    if not isinstance(payload.get("progress_path"),str):
        raise QueueError("formal bundle progress destination is incomplete")
    progress=Path(payload["progress_path"])
    ensure_formal_directory(payload,progress.parent,"formal bundle progress parent")
    if progress.exists() and progress.is_symlink():
        raise QueueError("formal bundle progress path is a symlink")
    return canonical

def output_records(payload, job):
    root=attempt_directory(payload,job).resolve(); records=[]
    for expected in payload["expected_outputs"]:
        try: raw=Path(expected["path"])
        except (OSError, TypeError) as exc: raise PublicationUncertain("cannot resolve declared output path") from exc
        is_formal_checkpoint = (job.get("kind") in {"formal_train", "formal_bundle", "formal_bundle_segment"} and
                                expected.get("artifact_type") == "checkpoint" and
                                expected.get("semantic") == "formal_training")
        if raw.is_absolute() and not is_formal_checkpoint:
            raise ConclusiveOutputFailure("attempt outputs must use relative paths")
        try:
            artifact=raw.resolve() if raw.is_absolute() else (root/raw).resolve()
            if (not raw.is_absolute() and root not in artifact.parents) or not artifact.is_file(): raise ConclusiveOutputFailure("missing declared output")
            if artifact.is_symlink(): raise ConclusiveOutputFailure("attempt output cannot be a symlink")
            with artifact.open("rb") as handle: os.fsync(handle.fileno())
            records.append({"path":str(artifact),"checksum":hashlib.sha256(artifact.read_bytes()).hexdigest()})
        except OSError as exc: raise PublicationUncertain("cannot read declared output") from exc
    try:
        descriptor=os.open(root,os.O_DIRECTORY); os.fsync(descriptor); os.close(descriptor)
    except OSError as exc: raise PublicationUncertain("cannot sync attempt output directory") from exc
    translated=[{**expected,"path":str(Path(expected['path']).resolve() if Path(expected['path']).is_absolute() else (root/Path(expected['path'])).resolve())} for expected in payload["expected_outputs"]]
    try: JobQueue.validate_artifacts(translated,records)
    except (ArtifactReadUncertain, CheckpointReadUncertain) as exc: raise PublicationUncertain(str(exc)) from exc
    except OSError as exc: raise PublicationUncertain("cannot inspect declared output metadata") from exc
    except QueueError as exc: raise ConclusiveOutputFailure(str(exc)) from exc
    return records

def sync_attempt_publication(run_dir, records, completion, checkpoint_root=None):
    """Durably retain all declared files and their ancestry before SQL success."""
    root=Path(run_dir).resolve()
    checkpoint_roots=[Path(value).resolve() for value in (() if checkpoint_root is None else checkpoint_root if isinstance(checkpoint_root,(list,tuple)) else (checkpoint_root,))]
    paths=[Path(record["path"]) for record in records] + [Path(completion)]
    for path in paths:
        with path.open("rb") as handle: os.fsync(handle.fileno())
        parent=path.parent.resolve()
        while True:
            descriptor=os.open(parent,os.O_DIRECTORY)
            try: os.fsync(descriptor)
            finally: os.close(descriptor)
            if parent == root or parent in checkpoint_roots: break
            if root not in parent.parents and not any(candidate in parent.parents for candidate in checkpoint_roots): raise QueueError("publication path escaped admitted roots")
            parent=parent.parent
    # Make the attempt directory itself reachable from its job/run parents.
    parent=root.parent
    # The worker may have created run_dir and arbitrary missing parents. Sync
    # all ancestor entries through the filesystem root before SQL success.
    while True:
        descriptor=os.open(parent,os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
        if parent == parent.parent: break
        parent=parent.parent

def commit_outputs(queue, job, payload, owner=None):
    if job.get("kind") == "prediction" and not any(item.get("artifact_type") == "prediction" and item.get("semantic") == "prediction" for item in payload.get("expected_outputs", [])):
        raise ConclusiveOutputFailure("prediction job lacks required prediction provenance contract")
    if job.get("kind") == "formal_train":
        outputs=payload.get("expected_outputs")
        if (not isinstance(outputs,list) or len(outputs)!=1 or outputs[0].get("path") != str((Path(payload["checkpoint_root"])/"final"/"manifest.json").resolve()) or
                outputs[0].get("semantic") != "formal_training" or outputs[0].get("run_identity") != payload.get("run_identity")):
            raise ConclusiveOutputFailure("formal training lacks the admitted final checkpoint contract")
        ensure_checkpoint_directory(payload)
    if job.get("kind") in {"formal_bundle", "formal_bundle_segment"}:
        ensure_bundle_checkpoint_directories(payload)
    run_dir=ensure_attempt_directory(payload, job)
    try: run_dir.mkdir(parents=True,exist_ok=True)
    except OSError as exc: raise PublicationUncertain("cannot create attempt publication directory") from exc
    temporary, final=run_dir / "result.tmp", run_dir / "result.json"
    records=output_records(payload,job)
    completion=run_dir / "producer_completion.json"
    try: produced=json.loads(completion.read_text())
    except FileNotFoundError as exc: raise ConclusiveOutputFailure("missing producer completion") from exc
    except OSError as exc: raise PublicationUncertain("cannot read producer completion") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc: raise ConclusiveOutputFailure("missing valid attempt-bound producer completion") from exc
    if not isinstance(produced,dict): raise ConclusiveOutputFailure("producer completion is not an object")
    if (produced.get("job_key") != job["job_key"] or produced.get("lease_token") != job["lease_token"] or
            produced.get("input_hash") != job.get("input_hash") or produced.get("artifacts") != records):
        raise ConclusiveOutputFailure("producer completion does not bind this attempt and inputs")
    if job.get("kind") in {"formal_train", "formal_bundle", "formal_bundle_segment"}:
        queue.completion_guard(job["job_key"],job["lease_token"])
    try:
        with completion.open("rb") as handle: os.fsync(handle.fileno())
    except OSError as exc: raise PublicationUncertain("cannot sync producer completion") from exc
    # Formal checkpoint has an implicit payload that must be durable too.
    payload_records=list(records)
    for record in records:
        if Path(record["path"]).name == "manifest.json":
            state=Path(record["path"]).parent / "state.pt"
            if state.is_file():
                try: payload_records.append({"path":str(state),"checksum":hashlib.sha256(state.read_bytes()).hexdigest()})
                except OSError as exc: raise PublicationUncertain("cannot read checkpoint payload for publication") from exc
    checkpoint_roots = (payload.get("checkpoint_root") if job.get("kind") == "formal_train" else
                        payload.get("checkpoint_roots") if job.get("kind") in {"formal_bundle", "formal_bundle_segment"} else None)
    try: sync_attempt_publication(run_dir, payload_records, completion, checkpoint_roots)
    except OSError as exc: raise PublicationUncertain("cannot durably sync attempt publication") from exc
    durable = temporary if temporary.exists() else final if final.exists() else None
    if durable is not None:
        try: published=json.loads(durable.read_text())
        except OSError as exc: raise PublicationUncertain("cannot read durable result record") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc: raise ConclusiveOutputFailure("invalid durable result record") from exc
        if not isinstance(published,dict): raise ConclusiveOutputFailure("durable result record is not an object")
        if published.get("job_key") != job["job_key"] or published.get("lease_token") != job["lease_token"] or published.get("artifacts") != records:
            raise ConclusiveOutputFailure("durable result record does not match current attempt")
    else:
        try: write_journal(temporary, {"job_key":job["job_key"],"lease_token":job["lease_token"],"artifacts":records,"completed_at":time.time()})
        except OSError as exc: raise PublicationUncertain("cannot write durable result record") from exc
    try: queue.commit(job["job_key"],job["lease_token"],temporary,final,owner=owner)
    except HardLimit:
        raise
    except (OSError, sqlite3.Error, RuntimeError, QueueError) as exc: raise PublicationUncertain("atomic output publication failed") from exc

def durable_publication_exists(payload, job):
    run=attempt_directory(payload,job)
    return any((run/name).is_file() for name in ("producer_completion.json", "result.tmp", "result.json"))

def progress_timeout(payload):
    p99 = payload.get("progress_p99_seconds")
    if not isinstance(p99, (int, float)) or p99 <= 0: return 1800
    return max(1800, 5 * p99)

def monitor(queue, job, process, payload, owner):
    """Heartbeat proves ownership; progress timeout catches a stalled live child."""
    while process.poll() is None:
        members=group_members(process.pid)
        if members is None: raise QueueError("process group observation unknown")
        process.known_members.update(members)
        if getattr(process,"journal_path",None):
            queue.update_attempt_journal(job["job_key"],job["lease_token"],process.pid,job["process_starttime"],str(payload.get("physical_gpu")),process.journal_path,member_identities=process.known_members)
        time.sleep(poll_seconds(payload))
        observe_progress(queue, job, payload.get("progress_path"), owner); queue.runtime_guard(job["job_key"], job["lease_token"])
        enforce_progress_timeout(queue, job, payload)
        queue.heartbeat(job["job_key"], job["lease_token"], pid=process.pid, process_starttime=process_starttime(process.pid), owner=owner)

def enforce_progress_timeout(queue, job, payload):
    row = next(row for row in queue.status() if row["job_key"] == job["job_key"])
    last_progress = row.get("progress_at") or queue.attempt_started_at(job["job_key"], job["lease_token"])
    if last_progress and time.time() - last_progress > progress_timeout(payload):
        # A stale counter alone is not permission to terminate normal long work.
        queue.protected_live(job["job_key"], job["lease_token"], "suspected progress stall; verify long-input work before termination", owner=job.get("lease_owner"))

def adopt_live(queue, owner):
    """Reattach after controller death; never launch a second same-attempt child."""
    host = owner.split(":", 1)[0]
    for job in queue.running_attempts():
        if job.get("lease_owner", "").split(":", 1)[0] != host: continue
        if job.get("state") == "LAUNCHING":
            try: repair_lock=acquire_supervisor_lock(attempt_directory(json.loads(job["payload"]),job))
            except OSError: continue
            if repair_lock is False: continue
            try:
                expected_journal=attempt_directory(json.loads(job["payload"]),job) / f"attempt-{job['attempts']}.json"
                if not queue.repair_launching_journal(job["job_key"],job["lease_token"],job["attempts"],expected_journal):
                    continue
                job=next(candidate for candidate in queue.running_attempts() if candidate["job_key"] == job["job_key"] and candidate["lease_token"] == job["lease_token"])
            finally:
                repair_lock.close()
        state=attempt_state(job.get("pid"), job.get("process_starttime"), host)
        if state not in {"live", "group_live"}: continue
        try: lock=acquire_supervisor_lock(attempt_directory(json.loads(job["payload"]), job))
        except OSError: continue
        if lock is False: continue
        queue.take_supervision(job["job_key"], job["lease_token"], owner)
        job=next(candidate for candidate in queue.running_attempts() if candidate["job_key"] == job["job_key"] and candidate["lease_token"] == job["lease_token"])
        job["lease_owner"] = owner
        state=attempt_state(job.get("pid"), job.get("process_starttime"), host)
        if state not in {"live", "group_live"}: lock.close(); continue
        try: known={int(k):str(v) for k,v in json.loads(Path(job["journal_path"]).read_text()).get("member_identities",{}).items()}
        except (OSError, ValueError, TypeError): known={}
        if state == "group_live" and not known: lock.close(); continue
        leader_before=process_starttime(job["pid"]) if state == "live" else None
        current=group_members(job["pid"])
        # A verified live leader authenticates newly spawned descendants; once
        # it is gone, only the prior durable snapshot can establish continuity.
        if (current is None or (state == "live" and leader_before != job["process_starttime"]) or
                (state == "live" and process_starttime(job["pid"]) != leader_before) or
                (state == "group_live" and not set(current.items()) <= set(known.items()))): lock.close(); continue
        known.update(current)
        if job.get("kind") in {"formal_train", "formal_bundle", "formal_bundle_segment"}:
            candidates=([job["pid"]] if state == "live" else list(current))
            inherited_lock=False
            for pid in candidates:
                try:
                    queue.recovered_lock_guard(job["job_key"],job["lease_token"],pid)
                    inherited_lock=True; break
                except QueueError:
                    continue
            if not inherited_lock:
                lock.close(); continue
        payload=json.loads(job["payload"])
        queue.update_attempt_journal(job["job_key"],job["lease_token"],job["pid"],job["process_starttime"],str(payload.get("physical_gpu")),job["journal_path"],member_identities=known)
        process = type("Adopted", (), {"pid":job["pid"], "group_only":state == "group_live", "known_members":known, "journal_path":Path(job["journal_path"]), "lock":lock, "poll":lambda self: None})()
        # The controller cannot obtain a child's return code after adoption; its
        # durable output contract decides success once the PID exits.
        return job, payload, process
    return None

def reconcile_exited_attempts(queue, owner):
    """Finish a durable child outcome after its supervising controller died."""
    host=owner.split(":",1)[0]
    for job in queue.running_attempts():
        if job.get("lease_owner", "").split(":",1)[0] != host: continue
        if job.get("state") == "LAUNCHING":
            try: lock=acquire_supervisor_lock(attempt_directory(json.loads(job["payload"]),job))
            except OSError: continue
            if lock is False: continue
            expected_journal=attempt_directory(json.loads(job["payload"]),job) / f"attempt-{job['attempts']}.json"
            if not queue.repair_launching_journal(job["job_key"],job["lease_token"],job["attempts"],expected_journal):
                queue.take_supervision(job["job_key"],job["lease_token"],owner); job["lease_owner"]=owner
                queue.protected_live(job["job_key"], job["lease_token"], "launch intent has no durable child identity", owner=owner)
                lock.close()
                continue
            job=next(candidate for candidate in queue.running_attempts() if candidate["job_key"] == job["job_key"] and candidate["lease_token"] == job["lease_token"])
        else:
            lock=None
        state = attempt_state(job.get("pid"), job.get("process_starttime"), host)
        if state == "live":
            if lock is not None: lock.close()
            continue
        if state == "group_live":
            if lock is not None: lock.close()
            continue
        if lock is None:
            try: lock=acquire_supervisor_lock(attempt_directory(json.loads(job["payload"]),job))
            except OSError: continue
            if lock is False: continue
        queue.take_supervision(job["job_key"],job["lease_token"],owner); job["lease_owner"]=owner
        # The job row is policy authority. Re-read it after ownership transfer
        # so a stop written by a prior controller cannot be bypassed by this
        # controller's pre-lock snapshot.
        job=next(candidate for candidate in queue.running_attempts() if candidate["job_key"] == job["job_key"] and candidate["lease_token"] == job["lease_token"])
        if state in {"foreign", "unknown", "identity_mismatch"}:
            queue.protected_live(job["job_key"], job["lease_token"], "PID identity mismatch; child outcome unknown", owner=owner)
            lock.close()
            continue
        if job.get("protective_stop_code"):
            queue.fail(job["job_key"],job["lease_token"],f"hard limit: {job['protective_stop_code']}",owner=owner,code=job["protective_stop_code"])
            lock.close(); continue
        payload=json.loads(job["payload"])
        try:
            commit_outputs(queue,job,payload,owner)
        except HardLimit as error:
            queue.fail(job["job_key"],job["lease_token"],f"hard limit: {error.code}",owner=owner,code=error.code)
        except ConclusiveOutputFailure as error:
                queue.fail(job["job_key"],job["lease_token"],f"exited child reconciliation: {error}",owner=owner)
        except PublicationUncertain as error:
            queue.protected_live(job["job_key"],job["lease_token"],f"reconciliation publication uncertain: {error}",owner=owner)
        except OSError as error:
            queue.protected_live(job["job_key"],job["lease_token"],f"reconciliation publication uncertain: {error}",owner=owner)
        finally:
            lock.close()

def worker(queue, owner, once):
    while True:
        reconcile_exited_attempts(queue,owner); queue.recover_expired(); adopted=adopt_live(queue, owner)
        if adopted:
            job,payload,process=adopted
            try:
                if job.get("protective_stop_code"):
                    code=job["protective_stop_code"]
                    if not terminate_group(process.pid, job["process_starttime"], process.known_members):
                        queue.protected_live(job["job_key"],job["lease_token"],f"protective stop retained pending verified exit: {code}",owner=owner)
                        return 1
                    queue.fail(job["job_key"],job["lease_token"],f"hard limit: {code}",owner=owner,code=code)
                    if once: return 1
                    continue
                while group_state(process.pid) == "live":
                    members=group_members(process.pid)
                    if members is None: raise QueueError("process group observation unknown")
                    # While the leader identity remains verified, descendants
                    # become part of this attempt's continuity snapshot.
                    if process_live(process.pid, job["process_starttime"], owner.split(":",1)[0]):
                        process.known_members.update(members)
                        queue.update_attempt_journal(job["job_key"],job["lease_token"],process.pid,job["process_starttime"],str(payload.get("physical_gpu")),process.journal_path,member_identities=process.known_members)
                    time.sleep(poll_seconds(payload)); observe_progress(queue,job,payload.get("progress_path")); queue.runtime_guard(job["job_key"], job["lease_token"]); enforce_progress_timeout(queue, job, payload); queue.heartbeat(job["job_key"],job["lease_token"],pid=process.pid,process_starttime=job["process_starttime"],owner=owner)
            except Exception as exc:
                hard = isinstance(exc, HardLimit)
                if hard: queue.record_protective_stop(job["job_key"],job["lease_token"],exc.code,owner=owner)
                if not hard or not terminate_group(process.pid, job["process_starttime"], getattr(process,"known_members",None)):
                    queue.protected_live(job["job_key"], job["lease_token"], f"protective adopted-child failure: {exc}", owner=owner)
                    # A supervisor must restart us; never signal success while
                    # leaving a child that requires continued supervision.
                    return 1
                queue.fail(job["job_key"], job["lease_token"], str(exc),owner=owner,code=exc.code if isinstance(exc,HardLimit) else "transient")
                if once: return 1
                continue
            process.lock.close()
            if job.get("kind") in {"formal_train", "formal_bundle", "formal_bundle_segment"} and group_state(process.pid) == "gone":
                try: queue.record_terminal_evidence(job["job_key"],job["lease_token"],process.pid,job["process_starttime"])
                except (QueueError, HardLimit):
                    return 1
            reconcile_exited_attempts(queue,owner)
            if group_state(process.pid) != "gone":
                return 1
            if once: return 0
            continue
        job = queue.claim(owner)
        if not job:
            if once: return 0
            time.sleep(60); continue
        payload = json.loads(job["payload"]); command = payload["command"]
        if "run_dir" not in payload: raise QueueError("worker payload requires data-volume run_dir")
        run_dir = ensure_attempt_directory(payload, job)
        if job.get("kind") == "formal_train": ensure_checkpoint_directory(payload)
        if job.get("kind") in {"formal_bundle", "formal_bundle_segment"}: ensure_bundle_checkpoint_directories(payload)
        temporary = run_dir / "result.tmp"; final = run_dir / "result.json"
        process = None; supervisor = None; journal_path = run_dir / f"attempt-{job['attempts']}.json"
        try:
            supervisor=acquire_supervisor_lock(run_dir)
            if supervisor is False: raise QueueError("attempt supervisor busy")
            with gpu_lock(payload.get("data_volume", queue.path.parent), payload.get("physical_gpu")) as lock_handle:
                if lock_handle is False:
                    queue.release_unstarted(job["job_key"], job["lease_token"], "GPU lock busy")
                    if once: return 0
                    continue
                # The lock closes the local-device race. Revalidate every
                # immutable attestation binding immediately before Popen.
                queue.bind_reservation(job["job_key"], job["lease_token"], lock_handle)
                queue.launch_guard(job["job_key"], job["lease_token"], lock_handle)
                if job.get("kind") == "formal_train":
                    command, captured_environment, cuda_device=capture_formal_inputs(payload, run_dir)
                elif job.get("kind") in {"formal_bundle", "formal_bundle_segment"}:
                    command, captured_environment, cuda_device=capture_formal_bundle_inputs(payload, run_dir)
                # This durable intent closes the crash window before Popen. A
                # restarted worker protects it rather than risking a duplicate.
                queue.start_attempt_journal(job["job_key"], job["lease_token"], None, None, str(payload.get("physical_gpu")), journal_path, state="LAUNCHING")
                if job.get("kind") in {"formal_train", "formal_bundle", "formal_bundle_segment"}:
                    environment=captured_environment
                else:
                    environment=dict(os.environ)
                    cuda_device=str(payload["physical_gpu"])
                environment["CUDA_VISIBLE_DEVICES"]=cuda_device
                environment["NC_RTED_PRODUCER_COMPLETION"]=str(run_dir / "producer_completion.json")
                environment["NC_RTED_JOB_KEY"]=job["job_key"]; environment["NC_RTED_LEASE_TOKEN"]=job["lease_token"]; environment["NC_RTED_INPUT_HASH"]=job["input_hash"]
                if job.get("kind") in {"formal_bundle", "formal_bundle_segment"}:
                    environment["NC_RTED_PROGRESS_ROOT"]=str(run_dir)
                    environment["NC_RTED_PROGRESS_PATH"]=payload["progress_path"]
                process = subprocess.Popen(command, cwd=run_dir, start_new_session=True, env=environment, pass_fds=(() if lock_handle is None else (lock_handle.fileno(),)))
                starttime=process_starttime(process.pid)
                if starttime is None: raise RuntimeError("child PID disappeared before identity capture")
                job["process_starttime"] = starttime
                process.known_members=group_members(process.pid)
                process.journal_path=journal_path
                queue.heartbeat(job["job_key"], job["lease_token"], pid=process.pid, process_starttime=starttime, owner=owner)
                queue.update_attempt_journal(job["job_key"], job["lease_token"], process.pid, starttime, str(payload.get("physical_gpu")), journal_path, member_identities=process.known_members)
                monitor(queue,job,process,payload,owner)
                if process.returncode: raise RuntimeError(f"command exited {process.returncode}")
                state=group_state(process.pid)
                if state != "gone":
                    raise QueueError("process group remains live" if state == "live" else "process group observation unknown")
                if job.get("kind") in {"formal_train", "formal_bundle", "formal_bundle_segment"}:
                    queue.record_terminal_evidence(job["job_key"],job["lease_token"],process.pid,job["process_starttime"])
                commit_outputs(queue,job,payload,owner)
        except Exception as exc:
            protective=isinstance(exc, HardLimit)
            if protective:
                # Persist the outcome before signalling so recovery cannot
                # promote a completion produced after the hard limit was seen.
                queue.record_protective_stop(job["job_key"],job["lease_token"],exc.code,owner=owner)
            if process is not None and group_state(process.pid) != "gone":
                if not protective or not terminate_group(process.pid, job.get("process_starttime"), getattr(process,"known_members",None)):
                    queue.protected_live(job["job_key"],job["lease_token"],f"child/process group retained after worker error: {exc}",owner=owner)
                    return 1
            if protective:
                queue.fail(job["job_key"], job["lease_token"], str(exc),owner=owner,code=exc.code)
                return 1
            if durable_publication_exists(payload,job):
                queue.protected_live(job["job_key"],job["lease_token"],f"durable completion requires reconciliation: {exc}",owner=owner)
                return 1
            queue.fail(job["job_key"], job["lease_token"], str(exc),owner=owner,code=exc.code if isinstance(exc,HardLimit) else "transient")
        finally:
            if supervisor not in (None, False): supervisor.close()
        if once: return 0

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--db", default="state/nc_rted/tasks.sqlite3")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("init"); sub.add_parser("status"); sub.add_parser("recover"); sub.add_parser("reconcile")
    ev = sub.add_parser("evidence"); ev.add_argument("name"); ev.add_argument("path"); ev.add_argument("--checksum")
    ev.add_argument("--schema", default=""); ev.add_argument("--input-code-hash", default=""); ev.add_argument("--accepted", action="store_true")
    gate = sub.add_parser("complete-gate"); gate.add_argument("job_key")
    wk = sub.add_parser("worker"); wk.add_argument("--owner", default=f"{os.uname().nodename}:{os.getpid()}"); wk.add_argument("--once", action="store_true")
    args = parser.parse_args(); queue = JobQueue(args.db)
    if args.action == "init": queue.register_matrix(); print(json.dumps({"registered": 28, "formal_runs": 12, "formal_jobs_claimable": False}))
    elif args.action == "status": print(json.dumps(queue.status(), indent=2))
    elif args.action == "recover":
        reconcile_exited_attempts(queue, f"{os.uname().nodename}:recover:{os.getpid()}")
        print(queue.recover_expired())
    elif args.action == "reconcile": print(queue.reconcile())
    elif args.action == "evidence": queue.add_evidence(args.name, args.path, args.checksum, args.schema, args.input_code_hash, args.accepted)
    elif args.action == "complete-gate": queue.complete_gate(args.job_key)
    else: return worker(queue, args.owner, args.once)
if __name__ == "__main__":
    try: raise SystemExit(main())
    except QueueError as exc: raise SystemExit(f"queue error: {exc}")
