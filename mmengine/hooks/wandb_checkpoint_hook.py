# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import wandb

from mmengine.registry import HOOKS
from .checkpoint_hook import CheckpointHook
from .hook import Hook


@HOOKS.register_module()
class WandbCheckpointHook(Hook):
    """Save the final model in wandb.

    Args:
    init_kwargs (dict, optional): wandb initialization
        input parameters.
        See `wandb.init <https://docs.wandb.ai/ref/python/init>` for
        details. Defaults to None.
    """

    def __init__(self, init_kwargs: Optional[dict] = None):
        self._init_kwargs = init_kwargs

    def after_train(self, runner) -> None:
        """Publish the checkpoint after training.

        Args:
            runner (Runner): The runner of the training process.
        """

        # Initialize wandb if necessary
        if wandb.run is None:
            runner.logger.info('WandbCheckpointHook: Initializing wandb')
            wandb.init(**self._init_kwargs)
        else:
            runner.logger.info(
                'WandbCheckpointHook: wandb is already initialized')

        # Get the latest checkpoint
        checkpoint_hook = next(
            (h for h in runner.hooks if isinstance(h, CheckpointHook)), None)

        # Save
        if checkpoint_hook is not None:
            runner.logger.info(
                f'Saving the last checkpoint to wandb, {checkpoint_hook.last_ckpt}'
            )
            wandb.save(checkpoint_hook.last_ckpt, policy='now')
        else:
            runner.logger.info(
                'Failed to find CheckpointHook, did not load the last checkpoint to wandb'
            )
