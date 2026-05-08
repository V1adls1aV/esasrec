# Тут будет девблог

```sh
uv sync
```

### Пробный запуск

```sh
python -m sasrec.train --dataset ml-1m --max_epochs 1 --save_dir checkpoints/ml-1m --resume_best
```

### Обучение

Дефолтные блоки – standard (s). Так же есть выбор между linear (l), mamba (m), mamba-noff (mnff) и fft.

```sh
python3 -m sasrec.train --dataset ml-20m --attn_types s,s,s,s --layers_mask 1100,25 --num_heads 4 --num_negatives 256 --batch_size 2048 --device cuda --max_length 200 --save_dir checkpoints/base
```

Можно открепить процесс от текущего терминала, чтобы обучать ночью, например (в `errors.log` печатаются ошибки)

```sh
nohup python3 -m sasrec.train --dataset ml-20m --attn_types s,mnff,s,mnff --layers_mask 1100,20 --num_heads 4 --num_negatives 256 --batch_size 512 --device cuda --max_length 200 --save_dir checkpoints/mamba-new > /dev/null 2> checkpoints/errors.log &
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

mamba (m) – Слой mamba вместо аттеншна, структура та же.  
mamba-noff (mnff) – отключает feed-forward блоки после слоя mamba, поскольку у нее есть свой механизм, похожий на ff.

Без рута пакеты иначе не поставишь

```sh
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

wget https://github.com/state-spaces/mamba/releases/download/v2.2.3/mamba_ssm-2.2.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

uv подтянет пакеты из корня проекта

```sh
uv sync --group mamba
```

```sh
nohup python3 -m sasrec.train --dataset ml-20m --attn_types m,m --layers_mask 1100,20 --num_heads 4 --num_negatives 256 --batch_size 512 --device cuda --max_length 200 --save_dir checkpoints/m-2 > /dev/null 2> checkpoints/errors_m2.log &

nohup python3 -m sasrec.train --dataset ml-20m --attn_types mnff,mnff --layers_mask 1100,20 --num_heads 4 --num_negatives 256 --batch_size 512 --device cuda --max_length 200 --save_dir checkpoints/mnff-2 > /dev/null 2> checkpoints/errors_mnff2.log &

nohup python3 -m sasrec.train --dataset ml-20m --attn_types m,m,m,m --layers_mask 1100,20 --num_heads 4 --num_negatives 256 --batch_size 512 --device cuda --max_length 200 --save_dir checkpoints/m-4 > /dev/null 2> checkpoints/errors_m4.log &

nohup python3 -m sasrec.train --dataset ml-20m --attn_types m,mnff,m,mnff --layers_mask 1100,20 --num_heads 4 --num_negatives 256 --batch_size 512 --device cuda --max_length 200 --save_dir checkpoints/m-mnff-m-mnff > /dev/null 2> checkpoints/errors_m_mnff_m_mnff.log &
```
