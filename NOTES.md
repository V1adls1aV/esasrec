# Тут будет девблог

```sh
uv sync
```

### Пробный запуск

```sh
python -m sasrec.train --dataset ml-1m --max_epochs 1 --save_dir checkpoints/ml-1m --resume_best
```

### Обучение

Можно передать набор аттеншн слоев через запятую (`--attn_types standard,linear`)

```sh
python3 -m sasrec.train --dataset ml-20m --num_blocks 2 --num_heads 8 --num_negatives 256 --batch_size 2048 --device cuda --max_length 200 --save_dir checkpoints/base
```

### Бенчмаркинг

По умолчанию модель перегоняется в onnx формат и запускается на gpu (иное можно указать флагами)

```sh
python3 -m sasrec.benchmark --num_blocks 2 --num_heads 8 --max_length 200 --mode latency
```

```sh
python3 -m sasrec.benchmark --num_blocks 2 --num_heads 8 --max_length 200 --mode throughput
```
