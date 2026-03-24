from data_utils import PQ3DModel


class AnchorPQ3DModel(PQ3DModel):
    def __init__(self, stage1_dir, stage2_dir, min_decision_num=None):
        super().__init__(stage1_dir, stage2_dir, min_decision_num=min_decision_num)
        from transformers import CLIPTextModel

        # Prefer local model cache first, then HF id.
        candidates = [
            "/home/zhaochaoyang/hf_models/clip-vit-large-patch14",
            "openai/clip-vit-large-patch14",
        ]
        last_err = None
        self.clip_text_model = None
        for model_name in candidates:
            try:
                self.clip_text_model = CLIPTextModel.from_pretrained(model_name)
                break
            except Exception as err:
                last_err = err
        if self.clip_text_model is None:
            raise RuntimeError(f"failed to load CLIPTextModel: {last_err}")
        self.clip_text_model.eval()
        self.clip_text_model.cuda()

