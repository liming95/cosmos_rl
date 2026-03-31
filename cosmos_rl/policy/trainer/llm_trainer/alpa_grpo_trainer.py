
from cosmos_rl.policy.trainer.base import TrainerRegistry
from cosmos_rl.policy.trainer.llm_trainer.grpo_trainer import GRPOTrainer

@TrainerRegistry.register(trainer_type="alpa_grpo")
class AlpaGRPOTrainer(GRPOTrainer):
    def step_training(
        self,
        rollouts: List[Rollout],
        current_step: int,
        total_steps: int,
        remain_samples_num: int,
        inter_policy_nccl: HighAvailabilitylNccl,
        is_master_replica: bool,
        do_save_checkpoint: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        # 1. preprocess rollouts (reason, sampled actions)
        # 2. get advance
        # 3. reference value (inclucing two part, VLM is LLM arch, and another part is diffusion arch)
        # 4. old value
        # 5. calculate loss based on advance , reference value and old value
        # 6. reduce loss
        # 7. backward
        # 8. return results 
        pass