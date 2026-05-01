# Тут будет девблог

```sh
uv sync
```

### Пробный запуск

```sh
python -m sasrec.train --dataset ml-1m --max_epochs 1 --save_dir checkpoints/ml-1m --resume_best
```

### Обучение

```sh
python3 -m sasrec.train --dataset ml-20m --attn_types standard,linear --num_heads 4 --num_negatives 256 --batch_size 2048 --device cuda --max_length 200 --save_dir checkpoints/base
```

### Бенчмаркинг

По умолчанию модель перегоняется в onnx формат и запускается на gpu (иное можно указать флагами)

```sh
python3 -m sasrec.benchmark --attn_types standard,linear --num_heads 4 --max_length 200 --mode latency
```

```sh
python3 -m sasrec.benchmark --attn_types standard,linear --num_heads 4 --max_length 200 --mode throughput
```
