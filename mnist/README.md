## requirements

```
Keras==2.1.6  # for dataset
click
torch==0.4.0
torchfile==0.1.0
torchvision==0.2.1
```

<details><summary>memo</summary>
実験設定はできるだけ config に書く.
細かな数字 (epochs とか) は commandline オプションでも渡せるようにしておく.
両方にある場合は後者を優先して使う.
オプションで設定する以上に大きな変更はブランチを切る.
残すべき実験結果は commit message に頑張る.
</details>
