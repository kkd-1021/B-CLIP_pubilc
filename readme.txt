3min使用默认采样方式
optimizer = torch.optim.Adam(model.classification_net.parameters(), lr=0.1, eps=1e-08)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.000001)

使用preprocessed脚本训练
使用epoch 24 chkp
val使用多次val取平均