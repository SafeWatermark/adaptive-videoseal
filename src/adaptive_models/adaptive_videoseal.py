from videoseal.models import Videoseal
from scene_detectors.base import SceneDetector
import torch
from torch.nn import functional as F
from scene_detectors.registry import detector_registry
import copy


class AdaptiveVideoseal(Videoseal):
    @classmethod
    def from_videoseal(cls, model) -> "AdaptiveVideoseal":
        """
        Convert a Videoseal instance to an AdaptiveVideoseal instance.

        This method creates a deep copy of the model and changes its class
        to AdaptiveVideoseal, preserving all state_dict parameters and buffers.

        Args:
            model: A Videoseal model instance to convert.

        Returns:
            An AdaptiveVideoseal instance with the same parameters as the input model.
        """
        new = copy.deepcopy(model)
        new.__class__ = cls
        return new

    def set_scene_detector(self, scene_detector: SceneDetector):
        self.scene_detector = scene_detector

    def set_logger(self, logger):
        self.logger = logger

    @staticmethod
    def _apply_video_mode(
        preds_w: torch.Tensor,
        total_frames: int,
        step_size: torch.Tensor,
        video_mode: str,
    ) -> torch.Tensor:
        """
        applies the selected video mode to expand predictions across frames
        args:
            preds_w (torch.Tensor): predictions for key frames [n, c, h, w]
            total_frames (int): total number of frames to generate
            step_size (torch.Tensor): number of frames between key frames
            video_mode (str): the video mode to use. can be one of "repeat", "alternate", "interpolate"
        returns:
            torch.Tensor: expanded predictions [total_frames, c, h, w]
        """
        # repeat each prediction for step_size frames
        if video_mode == "repeat":
            preds_w = torch.repeat_interleave(preds_w, step_size, dim=0)
        elif video_mode == "alternate":
            full_size = (total_frames,) + preds_w.shape[1:]  # f c h w
            full_preds = torch.zeros(full_size, device=preds_w.device)  # f c h w
            key_frame_ids = torch.cumsum(
                torch.cat(
                    [torch.tensor([0], device=step_size.device), step_size], dim=0
                ),
                dim=0,
            )[:-1]
            full_preds[key_frame_ids] = preds_w  # place preds_w [n c h w] in key frames
            preds_w = full_preds  # f c h w
        # TODO: Vectorize for speed
        elif video_mode == "interpolate":
            # interpolate between predictions
            full_size = (total_frames,) + preds_w.shape[1:]  # f c h w
            full_preds = torch.zeros(full_size, device=preds_w.device)  # f c h w
            # interpolation factors
            alphas = []
            for i, interval in enumerate(step_size[:-1]):
                if i < step_size[:-1].size(0) - 1:
                    alpha = 1 - torch.linspace(
                        0, 1, steps=interval + 1, device=preds_w.device
                    )
                    alphas.append(alpha[:-1])
                else:
                    alpha = 1 - torch.linspace(
                        0, 1, steps=interval, device=preds_w.device
                    )
                    alphas.append(alpha)
            if len(alphas) == 0:
                # if there is only 1 key frame, interpolation is inappropriate. Change to "repeat" mode
                preds_w = torch.repeat_interleave(preds_w, step_size, dim=0)
                return preds_w
            alphas = torch.cat(alphas, dim=0).view(-1, 1, 1, 1)
            # key frames and shifted key frames
            start_frames = torch.repeat_interleave(preds_w[:-1], step_size[:-1], dim=0)
            end_frames = torch.repeat_interleave(preds_w[1:], step_size[:-1], dim=0)
            # interpolate between key frames and shifted
            interpolated_preds = (
                alphas * start_frames + (1 - alphas) * end_frames
            )  # (f-1)//step c h w
            # fill the rest of the frames with the last ones
            last_start = len(interpolated_preds)
            full_preds[:last_start] = interpolated_preds
            full_preds[last_start:] = preds_w[
                -1
            ]  # use last prediction for remaining frames
            preds_w = full_preds  # f c h w

        return preds_w[:total_frames]

    @torch.no_grad()
    def embed(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor = None,
        is_video: bool = True,
        interpolation: dict = {
            "mode": "bilinear",
            "align_corners": False,
            "antialias": True,
        },
        lowres_attenuation: bool = False,
    ) -> dict:
        """
        Generates watermarked videos from the input images and messages (used for inference).
        Videos may be arbitrarily sized.
        """
        """
      Generates watermarked videos from the input images and messages (used for inference).
      Videos may be arbitrarily sized.
      """
        if not is_video:
            # fallback on parent class for batch of images
            return super().embed(imgs, msgs, interpolation, lowres_attenuation)
        if msgs is None:
            msgs = self.get_random_msg()  # 1 x k
        else:
            assert msgs.shape[0] == 1, "Message should be unique"

        # encode by chunk of cksz imgs, propagate the wm to spsz next imgs
        chunk_size = self.chunk_size  # n=cksz
        step_size = self.step_size  # spsz

        imgs_w = torch.zeros_like(imgs)

        for ii in range(0, len(imgs[::step_size]), chunk_size):
            nimgs_in_ck = min(chunk_size, len(imgs[::step_size]) - ii)
            start = ii * step_size
            end = start + nimgs_in_ck * step_size
            all_imgs_in_ck = imgs[start:end, ...]

            # interpolate
            all_imgs_res = all_imgs_in_ck.clone()
            if all_imgs_res.shape[-2:] != (self.img_size, self.img_size):
                all_imgs_res = F.interpolate(
                    all_imgs_res, size=(self.img_size, self.img_size), **interpolation
                )
            all_imgs_res = all_imgs_res.to(self.device)

            # calculate distance and thresholding
            if not hasattr(self, "scene_detector"):
                self.set_scene_detector(detector_registry["default"]())
            distinct_ids = self.scene_detector.detect_key_frames(all_imgs_res)
            base_ids = torch.arange(
                0, all_imgs_res.size(0), step_size, device=self.device
            )

            key_frame_ids = torch.cat([base_ids, distinct_ids])
            key_frame_ids = torch.unique(key_frame_ids)

            temp = torch.cat(
                [
                    key_frame_ids,
                    torch.tensor([all_imgs_res.size(0)], device=self.device),
                ]
            )
            repeats = temp[1:] - temp[:-1]

            imgs_res = all_imgs_res[key_frame_ids]

            # get deltas for the chunk, and repeat them for each frame in the chunk
            if self.embedder.yuv:  # take y channel only
                imgs_res = self.rgb2yuv(imgs_res)[:, 0:1]
            repeat_msgs = msgs.repeat(key_frame_ids.size(0), 1)

            preds_w = self.embedder(imgs_res, repeat_msgs.to(self.device))

            if hasattr(self, "logger"):
                self.logger.info(f"adaptive step size: {repeats}")

            preds_w = self._apply_video_mode(
                preds_w, len(all_imgs_in_ck), repeats, self.video_mode
            )

            # attenuate at low resolution if needed
            if self.attenuation is not None and lowres_attenuation:
                self.attenuation.to(all_imgs_res.device)
                hmaps = self.attenuation.heatmaps(all_imgs_res)
                preds_w = hmaps * preds_w

            # interpolate back
            preds_w = preds_w.to(imgs.device)
            if all_imgs_in_ck.shape[-2:] != (self.img_size, self.img_size):
                preds_w = F.interpolate(
                    preds_w, size=all_imgs_in_ck.shape[-2:], **interpolation
                )

            # attenuate at full resolution if needed
            if self.attenuation is not None and not lowres_attenuation:
                self.attenuation.to(all_imgs_in_ck.device)
                hmaps = self.attenuation.heatmaps(all_imgs_in_ck)
                preds_w = hmaps * preds_w

            # blend
            all_imgs_in_ck_w = self.blender(all_imgs_in_ck, preds_w)
            imgs_w[start:end, ...] = all_imgs_in_ck_w  # n 3 h w

        # clamp
        if self.clamp:
            imgs_w = torch.clamp(imgs_w, 0, 1)

        outputs = {
            "imgs_w": imgs_w,  # watermarked imgs: f 3 h w
            "msgs": msgs[0:1].repeat(len(imgs), 1),  # original messages: f k
        }
        return outputs
