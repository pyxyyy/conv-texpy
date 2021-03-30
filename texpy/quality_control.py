"""
Quality control
"""
from typing import List, NamedTuple, Optional


class QualityControlDecision:
    """
    A data structure to captures required information to update
    a workers qualification score.

    Fields:
        should_approve: should we approve this response or not?
        short_reason: a short reason for update (used when short on space)
        reason: a longer reason the explanation
        qualification_value: the exact value to set the qualification
        qualification_update: how much the qualification should be updated by

    NOTE: We do not subclass `NamedTuple` here because the JSON encoder
    serializes any subclasses of tuple as a list, and does not respect
    the custom serializer defined in util.py
    the custom default() function. Otherwise it will just serialize this
    as a list.
    """
    should_approve: bool
    short_reason: str
    reason: str
    qualification_value: Optional[int] = None
    qualification_update: Optional[int] = None

    def __init__(self, 
                 should_approve: bool,
                 short_reason: str,
                 reason: str,
                 qualification_value: Optional[int] = None,
                 qualification_update: Optional[int] = None):
        self.should_approve = should_approve
        self.short_reason = short_reason
        self.reason = reason
        self.qualification_value = qualification_value
        self.qualification_update = qualification_update

    def __repr__(self):
        return str(self.asdict())

    def asdict(self):
        return {
            "should_approve": self.should_approve,
            "short_reason": self.short_reason,
            "reason": self.reason,
            "qualification_value": self.qualification_value,
            "qualification_update": self.qualification_update,
            }


def generate_explanation(decisions: List[QualityControlDecision], char_limit: int = 0) -> str:
    """
    Generates an explanation given a list of decisions that will fit in the character limit.
    """
    ret = "\n".join(f"* {decision.reason}" for decision in decisions)

    if char_limit == 0 or len(ret) < char_limit:
        return ret

    ret = "\n".join(f"* {decision.short_reason}" for decision in decisions)
    if len(ret) < char_limit:
        return ret
    suffix = "... (for more details contact us)"

    ret = ret[:char_limit - len(suffix)] + suffix

    return ret


__all__ = ["QualityControlDecision", "generate_explanation"]
