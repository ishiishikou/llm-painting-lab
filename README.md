# llm-painting-lab

LLMに与えるプロンプト、ツール、状態保持、参照情報、反復フィードバックの設計によって、視覚的なアウトプット品質をどこまで引き上げられるかを検証する実験用リポジトリです。

目的はモデル同士を同一条件で公平にベンチマークすることではありません。モデル単体では上位モデルに届かない場合でも、実行環境側で十分な足場を与えることで、システム全体としてアウトプットをどこまで近づけられるかを確認します。

## 現在の標準実験: 描画道具v4

《真珠の耳飾りの少女》を題材に、完成画像だけでなく**描画途中も人が油彩で描いているように見えること**を目標にしています。

標準工程は次の6段階です。

1. **構図** — 顔中心、目線、口位置、肩、輪郭の目安を少数の入り・抜きのある線で置く
2. **大きな形** — 暖色系の下塗りと太い筆で人物、ターバン、衣服をブロックインする
3. **明暗の面** — 大きな色面を作った後、必要な境界だけ Mixer / Smudge でなじませる
4. **顔構造** — 顔の中心線、目線、口位置を基準に顔面の立体を組み直す
5. **細部** — 目、鼻、口、布、真珠、輪郭を細筆・Variable-width Brushで決める
6. **仕上げ** — Glaze、局所Mixer/Smudge、残差修正で色味とアクセントを整える

```text
地塗り
  ↓
構図
  ↓
下塗り + 大きな形
  ↓
明暗と色面 + 必要箇所だけ混色/ぼかし
  ↓
顔構造
  ↓
細部
  ↓
薄いグレーズ + 局所調整
```

背景は地塗りとして最初から置き、背景の細かな変化に動画時間を使いません。

## 描画道具の使い分けルール

道具は単に実装するだけでなく、工程ごとに「使う場面」と「使わない場面」を固定しています。

詳細は [`docs/painting-tool-rules.md`](docs/painting-tool-rules.md) を参照してください。

v4で追加した主要道具:

| 道具 | 役割 |
| --- | --- |
| `variableBrush` | 入り・抜きのある線、筆圧感 |
| `mixerBrush` | 下地色を拾いながら新しい色と混ぜる |
| `smudgeBrush` | 既存色を筆方向へ引きずってなじませる |
| `glaze` | 低透明度の色膜を薄く重ねる |
| `clipBox` | Mask / Alpha Lock 相当の描画範囲制御 |
| `dryBrush` | 下塗り、大きな形、かすれた太筆 |

`mixerBrush` と `smudgeBrush` は単純な Gaussian Blur ではありません。その時点のキャンバス上の色を読み取り、筆の移動に沿って色を拾う / 運ぶ処理を行います。

## 参考にした公開描画過程

描画工程の設計は、次の公開動画を参考にしています。映像そのものを複製するのではなく、共通する「描く順番」「筆の大きさ」「下塗りから細部へ進む考え方」を実装ルールへ落としています。

- Vermeer “Girl with a Pearl Earring” Oil Painting Process  
  https://www.youtube.com/watch?v=ZaNKTMAnli0
- Art Reproduction (Vermeer - The Girl with a Pearl Earring) Hand-Painted Step by Step  
  https://www.youtube.com/watch?v=_0G5yDXFFQE
- Cesar Santos — Painting Process of a Self Portrait in the Mirror, First Painting Stage in Oils  
  https://www.youtube.com/watch?v=AB1BB9qv7sA

参考動画から特に採用したのは、**drawing → underpainting → broad-brush block-in → value/color masses → blending where needed → facial structure → details → glaze/final accents** という流れです。

## v3からv4で変えたこと

v3では、付箋状の色パッチをやめて太い `dryBrush` と筆先移動を導入しました。v4ではさらに「色を置く」だけでなく「キャンバス上の色を扱う」方向へ進めています。

- 構図線・重要ディテールに `variableBrush` を使用
- 明暗工程で `mixerBrush` を使い、下地色を拾いながら混色
- 柔らかくしたい境界だけ `smudgeBrush` を使用
- 仕上げ工程だけ `glaze` を低透明度で使用
- `clipBox` で人物 / 顔の範囲を制限し、Mask / Alpha Lock 相当の挙動を追加
- Mixer / Smudge / Glaze の全域適用を禁止し、工程別ルールをCIでも検査
- リプレイ動画に使用中のブラシ名と対象を表示

## 生成方法

```bash
python tools/reference_to_brush_process.py reference.jpg \
  --output strokes.generated.json \
  --preview preview.png \
  --metrics metrics.json \
  --checkpoint-dir checkpoints \
  --width 864 \
  --height 1024 \
  --detail 18000 \
  --finish 10000
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

## 描画道具リプレイ

```bash
python tools/render_brush_motion_video.py strokes.generated.json \
  --output replay-painting-tools.mp4 \
  --seconds 90 \
  --fps 12 \
  --width 640
```

主要ストロークはフレーム間で徐々に伸び、現在の筆先位置と使用中の道具を表示します。Mixer / Smudge の途中状態もキャンバス色を参照して描画します。

## GitHub Actions

`reference-demo` ワークフローは公開されている参照画像を取得し、次を自動生成・検証します。

- `preview.png`
- `strokes.generated.json`
- `metrics.json`
- 6工程のチェックポイントPNG
- `replay-painting-tools.mp4`
- `painting-tool-rules.md`
- Webビューア一式
- 6工程が連続ブロックになっていること
- Mixer / Smudge / Glaze / Variable-width / Mask相当が実際に使用されていること
- 禁止工程で調整系ブラシを使っていないこと
- 最終品質指標が最低基準を満たすこと

成果物は `painting-tools-demo` artifact として保存します。参照画像自体はリポジトリへ保存しません。

## Webビューア

```bash
npm run serve
```

参照ガイド生成版:

```text
http://127.0.0.1:8000/?mode=generated
```

厳密モード:

```text
http://127.0.0.1:8000/
```

## 実験上の考え方

このリポジトリで確認したいのは「モデル単体の能力」ではなく、モデルと環境を合わせたシステム全体の能力です。

環境側で機械的な負担を引き受けつつ、モデルが構図、明暗、色、形状、評価、修正方針といった高レベル判断へ推論資源を使える状態を作ります。
