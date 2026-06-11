import torch

from gluemap.ff_inference.local_inference import LocalInference


class VGGTLocalInference(LocalInference):
    """Local inference for VGGT backbone."""

    def predict(self, batch: dict) -> dict:
        """Run the VGGT backbone on a batch of images.

        The model is called under bfloat16 autocast and the encoded pose is
        decoded into camera matrices via
        ``vggt.utils.pose_enc.pose_encoding_to_extri_intri``.

        Args:
            batch: Dict with ``"images"`` of shape ``(B, N, 3, H, W)``.

        Returns:
            The model's full prediction dict augmented with ``extrinsics`` of
            shape ``(B, N, 4, 4)`` and ``intrinsics`` of shape
            ``(B, N, 3, 3)``.
        """
        images = batch["images"].to(self.device).contiguous()

        with torch.cuda.amp.autocast(dtype=self.dtype):
            predictions = self.model(images)

        # Extract extrinsics and intrinsics from pose encoding
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            predictions["pose_enc"], image_size_hw=images.shape[-2:]
        )
        predictions["extrinsics"] = extrinsics
        predictions["intrinsics"] = intrinsics

        return predictions
