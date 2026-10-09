"""Review-required Stage2 cache adapter for the isolated paper worktree.

The trainer must construct this only after the cache configuration is approved.
It intentionally has no fallback to another dataset index.
"""
from contextlib import contextmanager


class FailClosedStage2DatasetMixin:
    """Mix into a worktree-only LazySupervisedDataset subclass."""

    stage2_cache = None

    def __getitem__(self, index):
        # The released dataset substitutes later samples after errors. Formal
        # Stage2 must preserve the frozen sample/order binding instead.
        return self._get_item(index)

    @contextmanager
    def cache_media_path(self, relative_path, request_index):
        if self.stage2_cache is None:
            raise RuntimeError("Stage2 cache resolver is not configured")
        with self.stage2_cache.acquire(relative_path, request_index) as path:
            yield str(path)

    def _bind_stage2_requests(self):
        for annotation in self.list_data_dict:
            relative = annotation.get("_reactvau_relative_video")
            if "video" in annotation and not isinstance(relative, str):
                raise RuntimeError("Stage2 loader lost the frozen relative video path")
            if relative is not None:
                annotation["_reactvau_request_index"] = self.stage2_cache.request_index_for(annotation)

    def process_video(self, video_file, data_anno, data_args):
        relative = data_anno.get("_reactvau_relative_video")
        index = data_anno.get("_reactvau_request_index")
        if not isinstance(relative, str) or not isinstance(index, int):
            raise RuntimeError("Stage2 request is not bound to frozen path/index")
        with self.cache_media_path(relative, index) as cached_path:
            return super().process_video(cached_path, data_anno, data_args)


def process_video_with_cache(dataset, original_process_video, video_file, data_anno, data_args, *, relative_path, request_index):
    """Invoke upstream process_video unchanged while the resolver lock is held."""
    with dataset.cache_media_path(relative_path, request_index) as cached_path:
        return original_process_video(cached_path, data_anno, data_args)
