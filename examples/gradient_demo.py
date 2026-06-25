"""
梯度与训练过程演示

模型: pred = w × input + b
损失: loss = (pred - label)²
目标: 从随机初始值训练到 w=3, b=1

两个样本:
  input=2, label=7   (因为 3×2+1=7)
  input=3, label=10  (因为 3×3+1=10)
"""

import torch

# ========== 初始化参数 ==========
w = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
lr = 0.01

# 训练数据
inputs = [2.0, 3.0]
labels = [7.0, 10.0]

print(f"目标: w=3.0, b=1.0")
print(f"初始: w={w.item():.3f}, b={b.item():.3f}")
print(f"学习率: {lr}")
print("=" * 60)

# ========== 逐样本训练 (SGD, batch_size=1) ==========
for epoch in range(30):
    for i in range(len(inputs)):
        x = torch.tensor([inputs[i]])
        y = torch.tensor([labels[i]])

        # 前向传播
        pred = w * x + b
        loss = (pred - y) ** 2

        # 反向传播
        loss.backward()

        # backward 之后 .grad 一定存在
        assert w.grad is not None and b.grad is not None

        # 打印梯度
        if epoch < 3:  # 只打印前3轮的详细过程
            print(f"\nEpoch {epoch+1}, 样本{i+1}: input={x.item()}, label={y.item()}")
            print(f"  pred = {w.item():.3f} × {x.item():.0f} + {b.item():.3f} = {pred.item():.3f}")
            print(f"  loss = ({pred.item():.3f} - {y.item():.0f})² = {loss.item():.3f}")
            print(f"  w.grad = {w.grad.item():.3f}  (∂loss/∂w = 2(pred-label)×input)")
            print(f"  b.grad = {b.grad.item():.3f}  (∂loss/∂b = 2(pred-label))")

        # 更新参数（手动实现 optimizer.step()）
        with torch.no_grad():
            w.data -= lr * w.grad
            b.data -= lr * b.grad

        # 清零梯度（手动实现 optimizer.zero_grad()）
        w.grad.zero_()
        b.grad.zero_()

        if epoch < 3:
            print(f"  更新后: w={w.item():.3f}, b={b.item():.3f}")

    # 每轮结束后打印状态
    if epoch >= 3:
        # 计算两个样本的总 loss（不反向传播）
        with torch.no_grad():
            total_loss = sum((w * inputs[i] + b - labels[i]) ** 2 for i in range(2))
        if epoch % 5 == 0 or epoch == 29:
            assert isinstance(total_loss, torch.Tensor)
            print(f"Epoch {epoch+1:2d}: w={w.item():.4f}, b={b.item():.4f}, total_loss={total_loss.item():.4f}")

print("=" * 60)
print(f"最终: w={w.item():.4f}, b={b.item():.4f}")
print(f"验证: {w.item():.2f}×2+{b.item():.2f} = {(w*2+b).item():.2f} (应为7)")
print(f"验证: {w.item():.2f}×3+{b.item():.2f} = {(w*3+b).item():.2f} (应为10)")
