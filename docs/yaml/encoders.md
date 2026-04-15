# Encoders

Encoder chuyển đổi observation (ảnh, vector, text) thành latent vector dùng bởi actor và critic.

## Chọn encoder nào?

| Loại | Trường hợp dùng | Ví dụ môi trường |
|---|---|---|
| `mlp` | Vector state (joint angles, velocities) | HalfCheetah, Pendulum, Isaac Lab |
| `cnn` | Ảnh pixel (RGB, depth) | CarRacing, Isaac Lab visual |
| `lstm` | Chuỗi thời gian, stacked frames | POMDPs, partially observable envs |
| `text` | Embedding ngôn ngữ/câu lệnh | Language-conditioned tasks |
| `resnet` | Visual obs — transfer từ ImageNet | Robotics, sim-to-real |
| `efficientnet` | Visual obs nhẹ hơn ResNet | Edge deployment, real-robot |
| `vit` | Visual obs — attention-based | High-resolution, long-range features |
| `huggingface` | Language obs — bất kỳ HF text model | Language-conditioned, NLP-RL tasks |
| `hf_vision` | Visual obs — bất kỳ HF vision model | ViT, Swin, ConvNeXt từ HF Hub |

## Các trường cấu hình

| Trường | Bắt buộc | Mô tả |
|---|---|---|
| `name` | ✓ | ID duy nhất, dùng trong `flows` |
| `type` | ✓ | `mlp`, `cnn`, `lstm`, `text`, hoặc registry key |
| `input_name` | **bắt buộc** khi tên encoder ≠ obs key | Key trong observation dict |
| `input_dim` | mlp/lstm | Số chiều input vector |
| `input_shape` | cnn | `[C, H, W]` |
| `latent_dim` | ✓ | Kích thước output latent |
| `layers` | ✓ | Cấu trúc layers |
| `aux_type` | | `autoencoder`, `contrastive`, `byol`, `vae`, `drq`, `spr`, `barlow` |
| `aux_latent_dim` | | Kích thước projection head |
| `use_momentum` | | `true` cho BYOL/contrastive momentum |
| `momentum_tau` | | EMA coefficient (0.99 thường dùng) |
| `recurrent` | | Wrap encoder bằng LSTM |
| `lstm_hidden` | | Kích thước hidden khi dùng LSTM |
| `frame_stack` | | Số frames được stack |

## `input_name` — routing observation vào encoder

Đây là trường quan trọng nhất cần hiểu đúng.

### Vấn đề

Môi trường trả về obs dict với key cố định, thường là `{"state": array}`.
Encoder lại có `name` riêng (ví dụ `actor_state_enc`, `critic_state_enc`).

Khi hai tên này **không khớp nhau**, SRL không biết dùng obs key nào để feed vào encoder nào — dẫn đến `KeyError`.

### Quy tắc routing (theo thứ tự ưu tiên)

| Ưu tiên | Điều kiện | Hành vi |
|---|---|---|
| 1 | `input_name` được khai báo | Dùng đúng key đó — **explicit, an toàn nhất** |
| 2 | Tên encoder == obs key | Passthrough trực tiếp |
| 3 | 1 obs key, 1 encoder | Tự động đổi tên |
| 4 | Số obs key == số encoder | Zip theo thứ tự |
| 5 | Fallback | Truyền nguyên dict |

Quy tắc 3 và 4 **không đáng tin** vì phụ thuộc vào thứ tự. Luôn dùng `input_name`.

### Pattern phổ biến: 2 encoder đọc cùng 1 obs key

Đây là pattern chuẩn khi actor và critic cần encoder riêng (không chia sẻ weights):

```yaml
# Env trả về: {"state": array[3]}

encoders:
  - name: actor_state_enc   # ← tên KHÁC với obs key "state"
    type: mlp
    input_name: state        # ← phải khai báo để router biết đọc từ "state"
    input_dim: 3
    latent_dim: 64

  - name: critic_state_enc  # ← tên KHÁC với obs key "state"
    type: mlp
    input_name: state        # ← cả hai encoder đều đọc cùng key "state"
    input_dim: 3
    latent_dim: 64

flows:
  - "actor_state_enc -> actor"
  - "critic_state_enc -> critic"
```

**Nếu bỏ `input_name`:** Router thấy 2 encoder names (`actor_state_enc`, `critic_state_enc`) và 1 obs key (`state`) — không khớp tên nào → `KeyError: 'actor_state_enc'`.

### Pattern đơn giản: 1 encoder, tên trùng obs key

Nếu chỉ có 1 encoder và đặt tên trùng với obs key, không cần `input_name`:

```yaml
# Env trả về: {"state": array[17]}

encoders:
  - name: state             # ← tên TRÙNG với obs key → tự động passthrough
    type: mlp
    input_dim: 17
    latent_dim: 256
```

### Multimodal: nhiều obs key khác nhau

```yaml
# Env trả về: {"image": array[3,84,84], "proprio": array[12]}

encoders:
  - name: visual_enc
    input_name: image       # ← đọc từ key "image"
    type: cnn
    input_shape: [3, 84, 84]
    latent_dim: 256

  - name: state_enc
    input_name: proprio     # ← đọc từ key "proprio"
    type: mlp
    input_dim: 12
    latent_dim: 64
```

---

## Ví dụ MLP (state-based)

```yaml
encoders:
  - name: state_enc
    type: mlp
    input_dim: 17
    latent_dim: 256
    layers:
      - {out_features: 256, activation: relu, norm: layer_norm}
      - {out_features: 256, activation: relu}
```

## Ví dụ CNN (vision)

```yaml
encoders:
  - name: image_enc
    type: cnn
    input_shape: [3, 84, 84]
    latent_dim: 256
    layers:
      - [32, 8, 4, relu]
      - [64, 4, 2, relu]
      - [64, 3, 1, relu]
```

## Ví dụ CNN với auxiliary reconstruction

```yaml
encoders:
  - name: image_enc
    type: cnn
    input_shape: [3, 96, 96]
    latent_dim: 256
    aux_type: autoencoder
    aux_latent_dim: 128
    layers:
      - [32, 8, 4, relu]
      - [64, 4, 2, relu]
      - [64, 3, 1, relu]
```

## Ví dụ đa phương thức (image + state)

```yaml
encoders:
  - name: image_enc
    type: cnn
    input_name: front_camera
    input_shape: [3, 84, 84]
    latent_dim: 128

  - name: state_enc
    type: mlp
    input_name: joint_states
    input_dim: 18
    latent_dim: 64
```

!!! tip
    Luôn khai báo `input_name` khi có nhiều encoder để tránh ambiguous routing.

## Encoder optimizer (v0.2.0)

Encoder có optimizer riêng biệt để tránh double-update:

```yaml
train:
  encoder_update_freq: 2           # cập nhật encoder mỗi 2 critic steps
  encoder_optimize_with_critic: true
  lr_encoder: 3e-4
```

| Optimizer | Tham số | Khi nào step |
|---|---|---|
| `critic_optimizer` | Q-head params | Mỗi gradient step |
| `actor_optimizer` | Actor head params | Mỗi gradient step |
| `encoder_optimizer` | Tất cả encoder params | Mỗi `encoder_update_freq` critic steps |

---

## Pre-trained Encoders

SRL hỗ trợ backbone pre-trained từ **torchvision** và **HuggingFace Transformers**.
Các encoder này được đăng ký vào registry — dùng key tương ứng trong `type:`.

### Cài đặt

```bash
# Vision (ResNet, EfficientNet, ViT)
pip install srl-rl[vision]      # hoặc: pip install torchvision

# Language (BERT, DistilBERT, ...)
pip install srl-rl[nlp]         # hoặc: pip install transformers

# Cả hai
pip install srl-rl[vision,nlp]
```

---

### `type: resnet`

ResNet-18, 34, 50, 101, 152 từ torchvision. Backbone được strip FC layer, output đi qua Linear + LayerNorm projection.

| `extra` field | Mặc định | Mô tả |
|---|---|---|
| `model_variant` | `resnet18` | `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152` |
| `pretrained` | `true` | `true` / `"IMAGENET1K_V1"` / `false` |
| `freeze_backbone` | `false` | Freeze toàn bộ backbone |
| `freeze_layers` | `null` | Freeze chỉ N layer đầu tiên |
| `normalize_input` | `true` | ImageNet mean/std normalization |

```yaml
encoders:
  - name: visual_enc
    type: resnet
    input_shape: [3, 224, 224]
    latent_dim: 256
    model_variant: resnet50
    pretrained: true
    freeze_backbone: true
```

!!! tip
    `freeze_backbone: true` khuyến nghị khi dataset RL nhỏ (< 1M steps). Chỉ train projection head.

---

### `type: efficientnet`

EfficientNet-B0 đến B7. Nhẹ hơn ResNet, phù hợp deployment trên robot thực.

| `extra` field | Mặc định | Mô tả |
|---|---|---|
| `model_variant` | `efficientnet_b0` | `efficientnet_b0` … `efficientnet_b7` |
| `pretrained` | `true` | `true` / `false` |
| `freeze_backbone` | `false` | Freeze toàn bộ backbone |
| `normalize_input` | `true` | ImageNet mean/std normalization |

```yaml
encoders:
  - name: visual_enc
    type: efficientnet
    input_shape: [3, 224, 224]
    latent_dim: 128
    model_variant: efficientnet_b3
    pretrained: true
    freeze_backbone: false
```

---

### `type: vit`

Vision Transformer từ torchvision. Dùng CLS-token làm representation.
`heads` layer được thay bằng `Identity` để lấy raw CLS embedding.

| `extra` field | Mặc định | Mô tả |
|---|---|---|
| `model_variant` | `vit_b_16` | `vit_b_16`, `vit_b_32`, `vit_l_16`, `vit_l_32` |
| `pretrained` | `true` | `true` / `false` |
| `freeze_backbone` | `false` | Freeze transformer encoder blocks |
| `freeze_layers` | `null` | Freeze N transformer blocks đầu tiên |
| `normalize_input` | `true` | ImageNet mean/std normalization |

```yaml
encoders:
  - name: visual_enc
    type: vit
    input_shape: [3, 224, 224]
    latent_dim: 256
    model_variant: vit_b_16
    pretrained: true
    freeze_backbone: true
    freeze_layers: 9        # freeze 9/12 blocks, fine-tune 3 cuối
```

!!! warning "Input size"
    ViT-B/16 và ViT-L/16 yêu cầu `input_shape: [3, 224, 224]` (patch size 16×16).
    ViT-B/32 và ViT-L/32 cũng chạy với 224×224.

---

### `type: huggingface`

Bất kỳ encoder-only model nào từ HuggingFace Hub (BERT, DistilBERT, RoBERTa, ALBERT, DeBERTa, …).

**Input:** integer tensor `[B, seq_len]` chứa token IDs.
Attention mask được tự động tạo từ vị trí non-padding (`obs != pad_token_id`).

| `extra` field | Mặc định | Mô tả |
|---|---|---|
| `model_name` | **bắt buộc** | HuggingFace model ID, ví dụ `"distilbert-base-uncased"` |
| `pooling` | `cls` | `"cls"` (CLS token) hoặc `"mean"` (mean pooling) |
| `freeze_backbone` | `true` | Freeze toàn bộ transformer weights |
| `pad_token_id` | `0` | Token ID dùng cho padding |

```yaml
encoders:
  - name: text_enc
    type: huggingface
    input_dim: 64           # max sequence length (seq_len)
    latent_dim: 128
    model_name: distilbert-base-uncased
    pooling: cls
    freeze_backbone: true
```

!!! tip
    `freeze_backbone: true` (default) rất quan trọng — language model lớn (110M+ params) sẽ làm chậm training RL nếu fine-tune toàn bộ.

---

### `type: hf_vision`

Bất kỳ vision model nào từ HuggingFace Hub — ViT, Swin Transformer, ConvNeXt, DeiT, BEiT, ResNet (HF version), …

**Input:** pixel tensor `[B, C, H, W]` (uint8 hoặc float). Output dimension được tự động phát hiện qua probe forward.

| `extra` field | Mặc định | Mô tả |
|---|---|---|
| `model_name` | **bắt buộc** | HuggingFace model ID, ví dụ `"google/vit-base-patch16-224"` |
| `freeze_backbone` | `true` | Freeze toàn bộ backbone weights |
| `normalize_input` | `true` | Áp dụng ImageNet mean/std normalisation |

```yaml
encoders:
  - name: visual_enc
    type: hf_vision
    input_shape: [3, 224, 224]
    latent_dim: 256
    model_name: google/vit-base-patch16-224
    freeze_backbone: true

  # Swin Transformer
  - name: swin_enc
    type: hf_vision
    input_shape: [3, 224, 224]
    latent_dim: 256
    model_name: microsoft/swin-tiny-patch4-window7-224
    freeze_backbone: true

  # ConvNeXt
  - name: convnext_enc
    type: hf_vision
    input_shape: [3, 224, 224]
    latent_dim: 128
    model_name: facebook/convnext-tiny-224
    freeze_backbone: false
```

!!! warning "Kích thước input"
    Hầu hết HuggingFace vision models yêu cầu input **224×224**. Thêm resize wrapper vào env nếu cần.

!!! tip "Phân biệt `vit` và `hf_vision`"
    - `type: vit` — dùng **torchvision** ViT (nhanh, offline, không cần HF token)
    - `type: hf_vision` — dùng **HuggingFace Hub** (nhiều architecture hơn, cần internet lần đầu)

---

### Kết hợp pre-trained với tính năng khác

Pre-trained encoders hoạt động hoàn toàn với tất cả tính năng hiện có:

**Momentum encoder (BYOL/CURL):**
```yaml
encoders:
  - name: visual_enc
    type: resnet
    input_shape: [3, 224, 224]
    latent_dim: 256
    model_variant: resnet50
    pretrained: true
    use_momentum: true      # wrap với MomentumEncoder (EMA)
    momentum_tau: 0.99
    aux_type: contrastive
```

**Recurrent wrapper:**
```yaml
encoders:
  - name: visual_enc
    type: resnet
    input_shape: [3, 224, 224]
    latent_dim: 256
    model_variant: resnet18
    pretrained: true
    recurrent: true         # wrap với LSTMEncoder
    lstm_hidden: 512
```

**Đa phương thức (image + text):**
```yaml
encoders:
  - name: visual_enc
    type: resnet
    input_shape: [3, 224, 224]
    latent_dim: 256
    model_variant: resnet50
    pretrained: true
    freeze_backbone: true

  - name: text_enc
    type: huggingface
    input_dim: 64
    latent_dim: 128
    model_name: distilbert-base-uncased
    freeze_backbone: true

flows:
  - "visual_enc -> actor"
  - "text_enc   -> actor"
  - "visual_enc -> critic"
  - "text_enc   -> critic"
```

---

## Xem thêm

- [Auxiliary Representation Learning](auxiliary.md)
- [Heads & Flows](heads_flows.md)
- [Training Block Reference](training_block.md)
