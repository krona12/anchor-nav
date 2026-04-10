from .vllm_adapter import VLLMDescriptionAdapter


class GAPModule:
    def __init__(self) -> None:
        self.adapter = VLLMDescriptionAdapter()

    def simplify_description(self, description: str) -> str:
        parsed = self.adapter.parse_description(description)
        target = parsed["target"].strip()
        anchor = parsed["anchor"].strip()
        room = parsed["room"].strip()

        if not target:
            raise RuntimeError(f"GAP parse error: empty target for description: {description}")

        if room and anchor:
            return f"{target} in the {room} near {anchor}"
        if room:
            return f"{target} in the {room}"
        if anchor:
            return f"{target} near {anchor}"
        return target
