# llm-painting-lab

LLMに与えるプロンプト、ツール、状態保持、参照情報、反復フィードバックの設計によって、視覚的なアウトプット品質をどこまで引き上げられるかを検証する実験用リポジトリです。

目的はモデル同士を同一条件で公平にベンチマークすることではありません。モデル単体では上位モデルに届かない場合でも、実行環境側で十分な足場を与えることで、システム全体としてアウトプットをどこまで近づけられるかを確認します。

## 現在の標準実験: 筆運びv3

《真珠の耳飾りの少女》を題材に、完成画像だけでなく**描画途中も人が油彩で描いているように見えること**を目標にしています。

現在は次の6工程を、後付けラベルではなく実際のストローク生成順として使います。

1. **構図** — 顔中心、目線、口位置、肩、輪郭の目安だけを少数線で置く
2. **大きな形** — 暖色系の下塗りを入れ、その上から太い筆で人物・ターバン・衣服をブロックインする
3. **明暗の面** — 光と影、主要な色面を大きめの筆致で整理する
4. **顔構造** — 顔の中心線、目線、口位置を基準に顔面の立体を組み直す
5. **細部** — 目、鼻、口、布、真珠、輪郭などを描き込む
6. **仕上げ** — レンダリング結果と参照画像の残差を見て、人物領域だけを追加修正する

```text
地塗り
  ↓
構図
  ↓
下塗り + 大きな形
  ↓
明暗と色面
  ↓
顔構造
  ↓
細部
  ↓
仕上げ
```

背景は地塗りとして最初から置き、背景の細かな変化に動画時間を使いません。

## 参考にした公開描画過程

描画工程の設計は、次の公開動画を参考にしています。映像そのものを複製するのではなく、共通する「描く順番」「筆の大きさ」「下塗りから細部へ進む考え方」を実装ルールへ落としています。

- Vermeer “Girl with a Pearl Earring” Oil Painting Process  
  https://www.youtube.com/watch?v=ZaNKTMAnli0
- Art Reproduction (Vermeer - The Girl with a Pearl Earring) Hand-Painted Step by Step  
  https://www.youtube.com/watch?v=_0G5yDXFFQE
- Cesar Santos — Painting Process of a Self Portrait in the Mirror, First Painting Stage in Oils  
  https://www.youtube.com/watch?v=AB1BB9qv7sA

参考動画から特に採用したのは、**drawing → underpainting → broad-brush block-in → value/color masses → facial structure → details → final accents** という流れです。

## v3で変えたこと

以前の版は工程名こそ人間らしくても、序盤が「付箋のような色パッチ」や格子状の更新に見える問題がありました。

v3では次を変更しています。

- 序盤の主要ブラシを `dryBrush` に変更
- 円形スタンプ・四角い色片によるブロックインを廃止
- 下塗りでは暖色系の低彩度ストロークを先に重ねる
- 本塗りは部位ごとに方向を揃えた、長く太い筆致を使う
- 明暗工程でも局所勾配だけに追従せず、顔・ターバン・衣服ごとの筆方向を優先する
- 細部・仕上げは人物外接範囲と強い前景判定の両方を満たす場所に限定し、背景ノイズを抑える
- リプレイ動画では1ストロークを瞬間表示せず、始点から終点まで途中状態を描画する
- 動画に描画中の筆先位置を表示する
- 総ストローク数比例ではなく工程ごとに再生時間を割り当てる

## 現在の本番結果

GitHub Actions上で生成した標準結果は次のとおりです。

| 工程 | 工程内ストローク数 | 累積 |
| --- | ---: | ---: |
| 構図 | 9 | 9 |
| 大きな形 | 429 | 438 |
| 明暗の面 | 2,109 | 2,547 |
| 顔構造 | 2,403 | 4,950 |
| 細部 | 18,000 | 22,950 |
| 仕上げ | 10,000 | 32,950 |

ブラシ内訳:

| ブラシ | 本数 |
| --- | ---: |
| `dryBrush` | 4,938 |
| `line` | 28,012 |

最終品質指標:

| 指標 | 最終値 |
| --- | ---: |
| MAE | 11.2056 |
| PSNR | 22.3308 dB |
| エッジ誤差 | 28.0660 |

画素忠実度だけを最大化した過去版より数値は低いですが、v3では**途中経過の自然さと筆運び**を優先しています。完成画像そのものをCanvasへ貼り付ける方式ではなく、最終画像は全ストロークから再構築します。

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

## 筆運びリプレイ

```bash
python tools/render_brush_motion_video.py strokes.generated.json \
  --output replay-brush-motion.mp4 \
  --seconds 90 \
  --fps 12 \
  --width 640
```

主要ストロークはフレーム間で徐々に伸び、現在の筆先位置も表示します。GitHub Actionsでは `fonts-noto-cjk` を導入して日本語の文字化けを防ぎます。

## GitHub Actions

`reference-demo` ワークフローは公開されている参照画像を取得し、次を自動生成します。

- `preview.png`
- `strokes.generated.json`
- `metrics.json`
- 6工程のチェックポイントPNG
- `replay-brush-motion.mp4`
- Webビューア一式

成果物は `brush-motion-demo` artifact として保存します。参照画像自体はリポジトリへ保存しません。

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

今後は、さらに少ない太筆で面をまとめること、顔ランドマークの精密化、Painter / Critic分離、複数候補からの選択、収束判定を検証します。
