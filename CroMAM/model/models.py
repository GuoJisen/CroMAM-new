import torch
import torch.nn as nn
import torchvision
from model.vision_transformer_encoder import SIB
from model.class_token_transformer_encoder import MFM


class MLP(nn.Module):
    def __init__(
            self,
            in_dim=512,
            out_dim=1,
            dropout=0.1):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(in_dim // 2, out_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = {
            'pred': self.fc(x)
        }
        return out


class CroMAM(nn.Module):
    def __init__(
            self,
            backbone='swin_transformer',
            pretrained=True,
            outcome_dim=1,
            feature_dim=512,
            dropout=0.1,
            output_features=False,
            device=torch.device("cpu"),
            branches=1,
            args=None
    ):
        super(CroMAM, self).__init__()

        self.output_features = output_features
        self.device = device
        self.branches = branches
        self.args = args
        self.dropout = self.args.dropout

        # initialize backbone model
        if isinstance(backbone, str):
            if pretrained:
                self.backbone1 = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.IMAGENET1K_V1)
                self.backbone2 = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.IMAGENET1K_V1)
            else:
                self.backbone1 = torchvision.models.swin_t(weights=None)
                self.backbone2 = torchvision.models.swin_t(weights=None)
            backbone_nlayers = 0
            model_dim = {'swin_transformer': {0: 768}}
            feature_dim = model_dim[backbone][backbone_nlayers]
        else:
            pass

        self.backbone1.head = nn.Identity()
        self.backbone2.head = nn.Identity()

        # if multiple GPUs
        # if torch.cuda.device_count() > 1:
        #     self.backbone1 = nn.DataParallel(self.backbone1, device_ids=[1])
        #     self.backbone2 = nn.DataParallel(self.backbone2, device_ids=[1])

        self.sib1 = SIB(dim=feature_dim, depth=2, heads=8, mlp_dim=feature_dim * 2, dim_head=feature_dim // 8,
                        dropout=0.0)
        self.sib2 = SIB(dim=feature_dim, depth=2, heads=8, mlp_dim=feature_dim * 2, dim_head=feature_dim // 8,
                        dropout=0.0)
        self.mfm1 = MFM(dim=feature_dim, depth=2, heads=8, mlp_dim=feature_dim * 2, dim_head=feature_dim // 8,
                          dropout=0.0)
        self.mfm2 = MFM(dim=feature_dim, depth=2, heads=8, mlp_dim=feature_dim * 2, dim_head=feature_dim // 8,
                          dropout=0.0)

        self.head = MLP(
            in_dim=feature_dim * self.branches,
            out_dim=outcome_dim,
            dropout=dropout
        )

    def forward(self, x1=None, x2=None, ppi=1):
        features1 = self.backbone1(x1)
        features2 = self.backbone2(x2)

        # 64*768 ->  8*8*768
        features1 = features1.view(features1.size(0) // ppi, ppi, -1)
        features2 = features2.view(features2.size(0) // ppi, ppi, -1)

        # 8*8*768 -> 8*1*768 , 8*8*768
        cls_token1, p_tokens1 = self.sib1(features1)
        cls_token2, p_tokens2 = self.sib2(features2)
        # Exchange class tokens
        class_token1 = self.mfm1(cls_token1, p_tokens2)
        class_token2 = self.mfm2(cls_token2, p_tokens1)

        # 8*1536
        cls_token = torch.cat((class_token1, class_token2), dim=1)

        out = self.head(cls_token)  # MLP
        if self.output_features:
            out['features'] = cls_token
        return out
