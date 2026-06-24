import torch
from torch import nn
from Prompt.CoOp import PromptMaker
from utils import encode_text_with_prompt_ensemble_equall

REAL_NAME = {'Bleeding': 'bleeding'}

# --- [DBT-Bleed] surgical domain-prompt path disabled (kept for reference) ---
# Domain-specific prompt templates were an alternative to the generic fill-in-the-blank
# approach. They are NOT used in the best-performing setting (prompt_style='default').
# Domain-specific prompt templates (bypass the generic fill-in-the-blank approach)
# DOMAIN_PROMPTS = {
#     'Bleeding': {
#         'normal': [
#             'clean surgical field with no blood',
#             'dry tissue without bleeding',
#             'bloodless operative area',
#             'clear and dry surgical site',
#             'residual blood with no active bleeding',
#             'surgical field with hemostasis',
#             'dried blood without active source',
#         ],
#         'abnormal': [
#             'active bleeding in surgical field',
#             'blood oozing from tissue',
#             'red blood pooling in operative area',
#             'hemorrhage during surgery',
#             'blood flowing from a vessel',
#             'surgical field covered in blood',
#             'red fluid leaking from tissue',
#         ],
#     },
# }

class DummyOptimizer:
    def step(self):
        pass  # Do nothing

    def zero_grad(self):
        pass  # Do nothing

class PromptChooser(nn.Module):
    def __init__(self,
                 clip_model,
                 args,
                 device
                 ):
        super(PromptChooser, self).__init__()
        self.text_mood = args.text_mood
        self.lr =  args.learning_rate
        # --- [DBT-Bleed] only used by the disabled surgical domain-prompt path ---
        # prompt_style = getattr(args, 'prompt_style', 'default')

        if self.text_mood == 'fix':
            print('fix text')
            with torch.cuda.amp.autocast(), torch.no_grad():
                self.text_features_fix = encode_text_with_prompt_ensemble_equall(clip_model, REAL_NAME[args.obj], device)
                self.text_optimizer = DummyOptimizer()
        else:
            # Build prompt string lists from the generic templates (prompt_style='default').
            # --- [DBT-Bleed] surgical domain-prompt path disabled (kept for reference) ---
            # if prompt_style == 'surgical' and args.obj in DOMAIN_PROMPTS:
            #     print(f'Using surgical domain prompts for {args.obj}')
            #     prompted_state_abnormal = DOMAIN_PROMPTS[args.obj]['abnormal']
            #     prompted_state_normal = DOMAIN_PROMPTS[args.obj]['normal']
            # else:
            #     if prompt_style == 'surgical':
            #         print(f'Warning: no surgical prompts for {args.obj}, falling back to default')
            prompt_abnormal = ['{}', 'active {}', 'visible {}', 'oozing {}', 'area of {}', 'significant {}', 'tissue {}']
            prompted_state_abnormal = [state.format(REAL_NAME[args.obj]) for state in prompt_abnormal]
            prompt_normal = ['no {}', 'zero {}', 'clean of {}', 'without {}', 'free of {}', 'clear of {}', 'no active {}']
            prompted_state_normal = [state.format(REAL_NAME[args.obj]) for state in prompt_normal]

            # Instantiate PromptMaker for abnormal prompts
            self.prompt_maker_abnormal = PromptMaker(
                prompts={'abnormal': prompted_state_abnormal},
                clip_model=clip_model,
                n_ctx=8,
                CSC=True,
                class_token_position=["end",  "front", "middle"]
            ).to(device)
            self.prompt_maker_abnormal.train()

            if self.text_mood == 'learnable_all':
                print('learnable all')

                # Instantiate PromptMaker for normal prompts
                self.prompt_maker_normal = PromptMaker(
                    prompts={'normal': prompted_state_normal},
                    clip_model=clip_model,
                    n_ctx=8,
                    CSC=True,
                    class_token_position=["end",  "front", "middle"]
                ).to(device)
                # Set both models to training mode
                self.prompt_maker_normal.train()

                # Define a single optimizer for both PromptMakers
                self.text_optimizer = torch.optim.Adam(
                    [
                        {'params': self.prompt_maker_normal.prompt_learner.parameters(), 'lr': self.lr},
                        {'params': self.prompt_maker_abnormal.prompt_learner.parameters(), 'lr': self.lr}
                    ],
                    betas=(0.5, 0.999)
                )

            else: # only learnable abnormal
                print('learnable abnormal')
                with torch.cuda.amp.autocast(), torch.no_grad():
                    self.text_features_normal = encode_text_with_prompt_ensemble_equall(clip_model, REAL_NAME[args.obj], device)[:,0].unsqueeze(1)
                    print(self.text_features_normal.shape)
                self.text_optimizer = torch.optim.Adam(
                    [{'params': self.prompt_maker_abnormal.prompt_learner.parameters(), 'lr': self.lr}],
                    betas=(0.5, 0.999)
                    )


    def forward(self):
        if self.text_mood == 'fix':
            return self.text_features_fix
        else:
            text_features_abnormal = self.prompt_maker_abnormal()  # [768,1]
            if self.text_mood == 'learnable_all':
                text_features_normal = self.prompt_maker_normal()  # [768,1]
                return torch.cat([text_features_normal, text_features_abnormal], dim=1)
            else:
                return torch.cat([self.text_features_normal, text_features_abnormal], dim=1)


    def save_prompt(self, save_dict):
        # Add text-related components based on text_mood
        if self.text_mood == 'fix':
            save_dict['text_features_fix'] = self.text_features_fix
        elif self.text_mood == 'learnable_all':
            save_dict['prompt_maker_normal'] = self.prompt_maker_normal.state_dict()
            save_dict['prompt_maker_abnormal'] = self.prompt_maker_abnormal.state_dict()
        else:  # only learnable abnormal
            save_dict['prompt_maker_abnormal'] = self.prompt_maker_abnormal.state_dict()

        return save_dict
