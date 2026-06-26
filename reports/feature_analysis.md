# Feature analysis

Báo cáo này kiểm tra các feature TF-IDF có trọng số lớn trong Logistic Regression và Linear SVM.
Mục tiêu là phát hiện spurious correlation: các từ quá chung bị học nhầm thành tín hiệu malicious hoặc benign.

## Custom stopwords

`a`, `about`, `am`, `an`, `and`, `are`, `as`, `at`, `be`, `been`, `being`, `by`, `can`, `could`, `did`, `do`, `does`, `for`, `from`, `had`, `has`, `have`, `having`, `how`, `i`, `if`, `in`, `into`, `is`, `it`, `its`, `me`, `my`, `of`, `on`, `or`, `our`, `shall`, `should`, `than`, `that`, `the`, `then`, `this`, `to`, `us`, `was`, `we`, `were`, `what`, `when`, `where`, `which`, `who`, `whom`, `whose`, `why`, `will`, `with`, `would`, `you`, `your`, `yours`

Các token security quan trọng như `ignore`, `previous`, `instructions`, `system`, `prompt`, `reveal`, `bypass`, `api`, `key`, `whoami`, `eval`, `exec` không nằm trong stopwords.

## logistic_regression

### Top malicious indicators

Với Logistic Regression/Linear SVM, trọng số dương nghiêng về malicious.

- `pwned`: 6.059017
- `run`: 4.013587
- `whoami`: 2.972029
- `say`: 2.698522
- `no`: 2.469967
- `all`: 2.469711
- `only`: 2.209563
- `ignore`: 2.057072
- `now`: 2.034245
- `p\w\n\e\d`: 2.014901
- `execute`: 1.965250
- `output`: 1.956043
- `instructions`: 1.904032
- `import`: 1.865373
- `sentence`: 1.821152
- `bypass`: 1.758546
- `print`: 1.718850
- `above`: 1.689893
- `following`: 1.687724
- `respond`: 1.578173

### Top benign indicators

Với Logistic Regression/Linear SVM, trọng số âm nghiêng về benign.

- `write`: -3.310287
- `largest`: -3.074132
- `compare`: -2.947353
- `create`: -2.860455
- `deutschland`: -2.648104
- `explain`: -2.548666
- `techniques`: -2.508026
- `generate`: -2.476451
- `capital`: -2.415425
- `germany`: -2.408464
- `cybersecurity`: -2.296700
- `computing`: -2.155663
- `learning`: -2.100229
- `history`: -2.034252
- `algorithms`: -2.000447
- `tips`: -1.983679
- `work`: -1.965409
- `development`: -1.963407
- `discussing`: -1.934939
- `storytelling`: -1.864710

### Suspected biased features

- Không phát hiện stopword trong top malicious/top benign features.

## linear_svm

### Top malicious indicators

Với Logistic Regression/Linear SVM, trọng số dương nghiêng về malicious.

- `pwned`: 2.401530
- `run`: 1.419140
- `say`: 1.301250
- `forget`: 1.111517
- `sie`: 1.094852
- `everything`: 1.071793
- `no`: 1.048553
- `all`: 1.006963
- `whoami`: 0.976341
- `following`: 0.917059
- `forget everything`: 0.907264
- `show`: 0.896461
- `expert`: 0.890859
- `respond`: 0.858041
- `without`: 0.846842
- `now`: 0.835616
- `above`: 0.829463
- `only`: 0.822319
- `before`: 0.791265
- `act`: 0.781034

### Top benign indicators

Với Logistic Regression/Linear SVM, trọng số âm nghiêng về benign.

- `largest`: -1.627302
- `germany`: -1.429510
- `deutschland`: -1.410488
- `corona`: -1.308124
- `europa`: -1.229751
- `ukraine`: -1.228679
- `capital`: -1.224801
- `currency`: -1.200232
- `cybersecurity`: -1.139691
- `write`: -1.112245
- `describe`: -1.111905
- `europe`: -1.094132
- `algorithms`: -1.079334
- `history`: -1.078981
- `time`: -1.069127
- `ist`: -1.068258
- `israel`: -1.066047
- `best`: -1.062652
- `storytelling`: -1.040391
- `berlin`: -1.032453

### Suspected biased features

- Không phát hiện stopword trong top malicious/top benign features.

## random_forest

### Top feature importances

Random Forest feature importance chỉ cho biết feature quan trọng, không nói trực tiếp feature nghiêng về malicious hay benign.

- `pwned`: 0.042677
- `write`: 0.023852
- `run`: 0.019355
- `create`: 0.014252
- `explain`: 0.013589
- `ignore`: 0.012418
- `compare`: 0.011917
- `generate`: 0.010512
- `c`: 0.010332
- `say`: 0.010108
- `whoami`: 0.009793
- `largest`: 0.009238
- `help`: 0.008950
- `all`: 0.008878
- `instructions`: 0.008110
- `help write`: 0.008037
- `techniques`: 0.007573
- `only`: 0.007410
- `capital`: 0.007148
- `no`: 0.007037

### Suspected biased features

- Không phát hiện stopword trong top malicious/top benign features.

## Giải thích ngắn

Spurious correlation xảy ra khi model học nhầm một từ phổ biến thành dấu hiệu tấn công chỉ vì từ đó xuất hiện lệch trong dataset train.
Custom stopwords giúp loại các từ quá chung khỏi vocabulary TF-IDF, còn feature analysis giúp kiểm tra thủ công xem model còn đang dựa vào tín hiệu nhiễu hay không.
