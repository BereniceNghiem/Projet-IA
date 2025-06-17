from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/test_bere")
for step in range(10):
    writer.add_scalar("test_metric/example", 0.8 + 0.01 * step, step)
writer.close()