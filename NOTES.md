# Тут будет девблог

```sh
uv sync
```

### Пробный запуск

```sh
python -m sasrec.train --dataset ml-1m --max_epochs 1 --save_dir checkpoints/ml-1m --resume_best
```

### Обучение

Дефолтные блоки – standard (s). Так же есть выбор между linear (l), mamba (m) и fft.

```sh
python3 -m sasrec.train --dataset ml-20m --attn_types s,s,s,s --layers_mask 1100,25 --num_heads 4 --num_negatives 256 --batch_size 2048 --device cuda --max_length 200 --save_dir checkpoints/base
```

### Бенчмаркинг

По умолчанию модель перегоняется в onnx формат и запускается на gpu (иное можно указать флагами)

```sh
python3 -m sasrec.benchmark --attn_types s,s --num_heads 4 --max_length 200 --mode latency
```

```sh
python3 -m sasrec.benchmark --attn_types s,s --num_heads 4 --max_length 200 --mode throughput
```

### Mamba

Без рута пакеты иначе не поставишь

```sh
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

wget https://github.com/state-spaces/mamba/releases/download/v2.2.3/mamba_ssm-2.2.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

uv подтянет пакеты из корня проекта

```sh
uv sync --group mamba
```

---

```sh
nohup python3 -m sasrec.train --dataset ml-20m --attn_types s,m,s,m --layers_mask 1100,20 --num_heads 4 --num_negatives 256 --batch_size 256 --device cuda --max_length 200 --save_dir checkpoints/mamba-smsm-200 > checkpoints/train200.log 2>&1 &
```

```sh
nohup python3 -m sasrec.train --dataset ml-20m --attn_types s,m,s,m --layers_mask 1100,20 --num_heads 4 --num_negatives 256 --batch_size 128 --device cuda --max_length 500 --save_dir checkpoints/mamba-smsm-500 > checkpoints/train500.log 2>&1 &
```

```sh
python3 -m sasrec.benchmark --attn_types s,m,s,m --num_heads 4 --max_length 500 --mode latency --no-onnx
```
