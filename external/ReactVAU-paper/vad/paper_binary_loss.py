"""ReactVAU main-paper Eq. 2; no full-vocabulary or EOS training loss."""
import torch
import torch.nn.functional as F


def decision_token_ids(tokenizer):
    yes = tokenizer.encode("Yes", add_special_tokens=False)
    no = tokenizer.encode("No", add_special_tokens=False)
    if len(yes) != 1 or len(no) != 1 or yes[0] == no[0]:
        raise ValueError("Paper binary target requires distinct single-token Yes and No")
    return yes[0], no[0]


def localized_binary_loss(logits, labels, yes_token_id, no_token_id):
    """Mean localized Yes/No negative log likelihood at first suffix token.

    logits[b, p-1] predicts labels[b, p]. All remaining vocabulary logits,
    including EOS, are outside the two-token probability in paper Eq. 2.
    labels remain in the model forward to preserve PaliGemma's prefix mask.
    """
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or logits.shape[0] == 0:
        raise ValueError("Expected nonempty [batch,time,vocab] logits and [batch,time] labels")
    if not (0 <= yes_token_id < logits.shape[-1] and
            0 <= no_token_id < logits.shape[-1] and yes_token_id != no_token_id):
        raise ValueError("Invalid Yes/No vocabulary indices")
    active = labels.ne(-100)
    if not bool(active.any(dim=1).all()):
        raise ValueError("Every example needs a decision target")
    first = active.to(torch.int64).argmax(dim=1)
    if bool(first.eq(0).any()):
        raise ValueError("Decision target needs a preceding prediction position")
    rows = torch.arange(logits.shape[0], device=logits.device)
    target = labels[rows, first]
    if not bool(((target == yes_token_id) | (target == no_token_id)).all()):
        raise ValueError("First suffix target must be Yes or No")
    decision_logits = logits[rows, first - 1][:, [no_token_id, yes_token_id]].float()
    if not bool(torch.isfinite(decision_logits).all()):
        raise ValueError("Nonfinite decision logits")
    return F.cross_entropy(decision_logits, (target == yes_token_id).long())
