# llm-painting-lab

LLMに与えるプロンプト、ツール、状態保持、参照情報、反復フィードバックの設計によって、視覚的アウトプット品質をどこまで引き上げられるかを検証する実験用リポジトリです。

目的はモデル同士を同一条件で公平にベンチマークすることではありません。モデル単体では上位モデルに届かない場合でも、実行環境側で足場を与えることで、システム全体としてアウトプットをどこまで近づけられるかを確認します。

## 現在の標準実験: 部位別方向場 v5

《真珠の耳飾りの少女》を題材に、完成画像だけでなく**描画途中も人が油彩で描いているように見えること**を目標にしています。

標準工程は6段階です。

1. **構図** — 顔中心、目線、口位置、肩、輪郭の目安を少数の線で置く
2. **大きな形** — 暖色下塗りと太い筆で人物、ターバン、衣服をブロックインする
3. **明暗の面** — 大きな色面を作り、必要な境界だけ Mixer / Smudge でなじませる
4. **顔構造** — 目・鼻・口・顎・頬の関係を組み直す
5. **細部** — 顔・ターバン・真珠など重要部を優先して解像度を上げる
6. **仕上げ** — Glaze、局所Mixer/Smudge、細筆で色味・エッジ・アクセントを整える

背景は地塗りとして最初から置き、背景の細かな変化に動画時間を使いません。

## v5の中心: ストローク方向場

v4では `flatBrush` / `dryBrush` の方向が似た斜め角度へ集まりやすく、広い面が「同じ棒線を並べている」ように見える問題がありました。

v5では方向を固定角度や単純ランダムで決めず、**工程 × 部位 × 形状 × 局所エッジ**から選びます。

| 部位 | 主方向 |
| --- | --- |
| 顔 | 額・頬・顎の面を分け、顔構造以降だけ目=横、鼻筋=縦、口=横を補助 |
| ターバン | 冠部の巻き方向と、右側の垂れ布の縦方向 |
| 衣服 | 重力方向を基準に、左右へ少し開く流れ |
| その他人物 | シルエット・局所エッジに沿う方向 |

序盤は形状方向を強くし、細部ほど参照画像の局所エッジ方向を強めます。小さな乱数は単調さを消す用途だけに使います。

広い筆の角度分布はCIでも検査し、1方向への過度な集中を禁止しています。

## 描画道具

| 道具 | 役割 |
| --- | --- |
| `flatBrush` | 大きな面、ブロックイン、明暗面 |
| `dryBrush` | 下塗り、かすれ、粗い面 |
| `variableBrush` | 入り・抜きのある線、構造、重要ディテール |
| `mixerBrush` | 下地色を拾いながら新しい色と混ぜる |
| `smudgeBrush` | 既存色を筆方向へ引きずる |
| `glaze` | 低透明度の色膜を重ねる |
| `line` | 目・口・鼻・輪郭・真珠などの決め |
| `clipBox` | Mask / Alpha Lock 相当の描画範囲制御 |

`mixerBrush` と `smudgeBrush` は単純な Gaussian Blur ではなく、その時点のキャンバス色を読み取り、筆の移動に沿って色を拾う / 運ぶ処理です。

詳細ルール:

- [`docs/painting-tool-rules.md`](docs/painting-tool-rules.md) — 工程 × 部位 × 道具 × 方向
- [`docs/painting-evaluation-rules.md`](docs/painting-evaluation-rules.md) — 完成画と途中工程の評価基準

## 細部依存の削減

v5では「最後に大量の細線で全画面を復元する」依存も下げています。

- detail: 18,000 → 14,000 を上限目安に変更
- finish: 10,000 → 7,000 を上限目安に変更
- 顔 / ターバン / 衣服 / その他へ重要度別の割当を導入
- detail / finish の前半は `variableBrush` を中心にし、極短い一定幅線だけで埋めない

目標は、**細部工程へ入る前の大きな形・明暗の時点で既に似ていること**です。

## 参考にした公開描画過程

描画工程の設計は次の公開動画を参考にしています。映像を複製するのではなく、描く順番・筆の大きさ・下塗りから細部へ進む考え方を実装ルールへ落としています。

- Vermeer “Girl with a Pearl Earring” Oil Painting Process  
  https://www.youtube.com/watch?v=ZaNKTMAnli0
- Art Reproduction (Vermeer - The Girl with a Pearl Earring) Hand-Painted Step by Step  
  https://www.youtube.com/watch?v=_0G5yDXFFQE
- Cesar Santos — Painting Process of a Self Portrait in the Mirror, First Painting Stage in Oils  
  https://www.youtube.com/watch?v=AB1BB9qv7sA

## 生成方法

v5の標準エントリポイントは `generate_direction_field_v5.py` です。

```bash
python tools/generate_direction_field_v5.py reference.jpg \
  --output strokes.generated.json \
  --preview preview.png \
  --metrics metrics.json \
  --checkpoint-dir checkpoints \
  --width 864 \
  --height 1024 \
  --detail 14000 \
  --finish 7000
```

生成されるチェックポイント:

```text
checkpoints/01-composition.png
checkpoints/02-silhouette.png
checkpoints/03-light-shadow.png
checkpoints/04-face-structure.png
checkpoints/05-detail.png
checkpoints/06-finish.png
```

## リプレイ

```bash
python tools/render_brush_motion_video.py strokes.generated.json \
  --output replay-direction-field.mp4 \
  --seconds 90 \
  --fps 12 \
  --width 640
```

主要ストロークはフレーム間で徐々に伸び、現在の筆先位置と使用中の道具を表示します。

## GitHub Actions

`reference-demo` は公開参照画像を取得し、次を自動生成・検証します。

- `preview.png`
- `strokes.generated.json`
- `metrics.json`
- 6工程のチェックポイントPNG
- `replay-direction-field.mp4`
- 道具ルール / 評価ルール
- 6工程が連続ブロックになっていること
- Mixer / Smudge / Glaze / Variable / Mask相当が実際に使用されること
- 禁止工程で調整系ブラシを使わないこと
- 広筆の方向が1角度に過度集中しないこと
- 完成品質が最低閾値を満たすこと

成果物は `direction-field-demo` artifact として保存します。参照画像自体はリポジトリへ保存しません。

## 実験上の考え方

このリポジトリで確認したいのは「モデル単体の能力」ではなく、モデルと環境を合わせたシステム全体の能力です。

環境側で機械的な負担を引き受けつつ、モデルが構図、明暗、色、形状、エッジ、評価、修正方針といった高レベル判断へ推論資源を使える状態を作ります。
