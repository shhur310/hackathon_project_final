"""
Generate notebooks/eda_master_v2.ipynb implementing all 10 prompts from ChatGPT analysis.
Run this script ONCE to produce the notebook.
"""
import json
from pathlib import Path

def md(text):
    return {"cell_type":"markdown","metadata":{},"source":text.splitlines(keepends=True)}
def code(text):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":text.splitlines(keepends=True)}

cells = [
    md("# 마스터 EDA v2 — ChatGPT 10-prompt 통합\n\n"
       "**원천**: ChatGPT 분석에서 도출된 10개 프롬프트 (나이/이력/시술 유형/배아·난자/이식 전략/불임 원인/donor/결측/interaction/leakage)\n\n"
       "각 § 셀이 하나의 프롬프트에 대응. 모든 분석은 **train 데이터만 사용**, test 데이터로는 schema 검증과 drift 점검만.\n\n"
       "**Leakage 원칙**:\n"
       "- target rate, target encoding, scaling, imputation 통계는 train fold 내부에서만\n"
       "- test에는 transform/reindex만, fit 금지\n"
       "- pd.get_dummies는 train 기준 컬럼으로 test reindex"),
    code("import sys; sys.path.insert(0, '..')\n"
         "import pandas as pd, numpy as np\n"
         "import matplotlib.pyplot as plt\n"
         "import seaborn as sns\n"
         "import warnings; warnings.filterwarnings('ignore')\n"
         "plt.rcParams['figure.dpi'] = 90\n"
         "sns.set_palette('Set2')\n\n"
         "TARGET = '임신 성공 여부'\n"
         "DATA = Path('../data') if (Path('../data')).exists() else Path('data')\n"
         "from pathlib import Path\n"
         "DATA = Path('../data') if (Path('..')/'data').exists() else Path('data')\n"
         "train = pd.read_csv(DATA/'train.csv')\n"
         "test = pd.read_csv(DATA/'test.csv')\n"
         "print(f'train: {train.shape}, test: {test.shape}')\n"
         "print(f'positive rate: {train[TARGET].mean():.4%}')\n"
         "age_order = ['만18-34세','만35-37세','만38-39세','만40-42세','만43-44세','만45-50세','알 수 없음']\n"
         "count_order = ['0회','1회','2회','3회','4회','5회','6회 이상']\n"
         "COUNT_INT = {c:i for i,c in enumerate(count_order)}"),

    # =====================================================
    md("## §1. 나이 효과 중심 EDA\n\n"
       "**가설** (메타분석 2025 Hum Reprod): female age는 가장 강한 predictor 중 하나"),
    code("# 1.1 시술 당시 나이 구간별 성공률 + 표본수\n"
         "g = train.groupby('시술 당시 나이')[TARGET].agg(n='count', success_rate='mean').reindex(age_order)\n"
         "g['n_pos'] = (g['n'] * g['success_rate']).astype(int)\n"
         "print(g.round(4))\n"
         "fig, ax = plt.subplots(figsize=(9,3.5))\n"
         "ax.bar(range(len(g)), g['success_rate'], color='#4a90e2')\n"
         "ax.set_xticks(range(len(g))); ax.set_xticklabels(g.index, rotation=20)\n"
         "ax.axhline(train[TARGET].mean(), color='red', ls='--')\n"
         "ax.set_title('나이 구간별 임신 성공률'); ax.set_ylabel('성공률')\n"
         "plt.tight_layout(); plt.show()"),
    code("# 1.2 나이 × 시술 유형 (IVF / DI)\n"
         "ix = train.groupby(['시술 당시 나이','시술 유형'])[TARGET].mean().unstack().reindex(age_order)\n"
         "print(ix.round(3))\n"
         "ix.plot.bar(figsize=(10,3.5), title='나이 × 시술 유형 성공률'); plt.xticks(rotation=20); plt.tight_layout(); plt.show()"),
    code("# 1.3 나이 × 이식된 배아 수\n"
         "ivf = train[train['시술 유형']=='IVF']\n"
         "ix = ivf.groupby(['시술 당시 나이','이식된 배아 수'])[TARGET].mean().unstack().reindex(age_order)\n"
         "import seaborn as sns\n"
         "fig, ax = plt.subplots(figsize=(9,3.5))\n"
         "sns.heatmap(ix[[c for c in [0.0,1.0,2.0,3.0] if c in ix.columns]], annot=True, fmt='.3f', cmap='RdYlGn', center=0.25, ax=ax)\n"
         "ax.set_title('나이 × 이식 배아 수 (Lawlor 2012 가설)'); plt.tight_layout(); plt.show()"),
    code("# 1.4 나이 × 배아 이식 경과일 (day3 vs day5 proxy)\n"
         "day_map = {2:'day0-2', 3:'day3', 4:'day3', 5:'day5+', 6:'day5+', 7:'day5+'}\n"
         "ivf['_transfer_day_bin'] = ivf['배아 이식 경과일'].map(day_map).fillna('NA/cancelled')\n"
         "ix = ivf.groupby(['시술 당시 나이','_transfer_day_bin'])[TARGET].mean().unstack().reindex(age_order)\n"
         "print(ix.round(3))"),
    code("# 1.5 donor age × 본인 age\n"
         "donor_only = train[train['난자 출처']=='기증 제공']\n"
         "if len(donor_only) > 100:\n"
         "    print('기증 난자 사용 그룹 (n={})'.format(len(donor_only)))\n"
         "    print('난자 기증자 나이별 성공률:')\n"
         "    print(donor_only.groupby('난자 기증자 나이')[TARGET].agg(['mean','count']).round(4))\n"
         "    ix = donor_only.groupby(['시술 당시 나이','난자 기증자 나이'])[TARGET].mean().unstack()\n"
         "    print('\\n시술 당시 나이 × 난자 기증자 나이:')\n"
         "    print(ix.round(3))"),
    md("**§1 후보 파생변수**: `age_ord`, `age_mid`, `age_35p`, `age_38p`, `age_43p`, `age_unknown`, age × 시술 유형, age × 이식된 배아 수, age × 배아 이식 경과일, age × 난자 출처"),

    # =====================================================
    md("## §2. 과거 시술·임신·출산 이력 EDA\n\n"
       "**가설** (McLernon model): 과거 이력은 prognosis를 반영"),
    code("# 2.1 횟수 컬럼 → 정수 변환 + 분포\n"
         "count_cols = ['총 시술 횟수','IVF 시술 횟수','DI 시술 횟수',\n"
         "              '총 임신 횟수','IVF 임신 횟수','DI 임신 횟수',\n"
         "              '총 출산 횟수','IVF 출산 횟수','DI 출산 횟수']\n"
         "for c in count_cols:\n"
         "    if c in train.columns:\n"
         "        train[c+'_int'] = train[c].map(COUNT_INT).fillna(-1)\n"
         "fig, axes = plt.subplots(3, 3, figsize=(13,7))\n"
         "for ax, c in zip(axes.flat, count_cols):\n"
         "    if c+'_int' in train.columns:\n"
         "        train[c+'_int'].value_counts().sort_index().plot.bar(ax=ax, color='#4a90e2')\n"
         "        ax.set_title(c, fontsize=9)\n"
         "plt.tight_layout(); plt.show()"),
    code("# 2.2 과거 임신 있음/없음, 과거 출산 있음/없음\n"
         "train['_prior_pregnancy_any'] = (train['총 임신 횟수_int'] >= 1).astype(int)\n"
         "train['_prior_live_birth_any'] = (train['총 출산 횟수_int'] >= 1).astype(int)\n"
         "train['_prior_pregnancy_no_birth'] = ((train['총 임신 횟수_int'] >= 1) & (train['총 출산 횟수_int']==0)).astype(int)\n"
         "for c in ['_prior_pregnancy_any','_prior_live_birth_any','_prior_pregnancy_no_birth']:\n"
         "    print(f'{c}: n={train[c].sum()}, LBR={train[train[c]==1][TARGET].mean():.4f} vs no={train[train[c]==0][TARGET].mean():.4f}')"),
    code("# 2.3 ratio: prior_live_birth / prior_treatment\n"
         "for num_col, den_col, name in [\n"
         "    ('총 출산 횟수_int','총 시술 횟수_int','live_birth_per_treatment'),\n"
         "    ('총 임신 횟수_int','총 시술 횟수_int','pregnancy_per_treatment'),\n"
         "    ('IVF 임신 횟수_int','IVF 시술 횟수_int','ivf_preg_per_ivf_tr'),\n"
         "]:\n"
         "    if num_col in train and den_col in train:\n"
         "        ratio = train[num_col] / train[den_col].replace(0, np.nan)\n"
         "        train['_'+name] = ratio.fillna(-1).clip(-1, 2)\n"
         "        print(f'{name}: corr with target = {train[\"_\"+name].corr(train[TARGET]):.4f}')"),
    md("**§2 후보 파생**: `count_int` (모든 횟수), `prior_pregnancy_any`, `prior_live_birth_any`, `prior_pregnancy_no_live_birth`, `live_birth_per_treatment`, `pregnancy_per_treatment`, IVF/DI별 ratio"),

    # =====================================================
    md("## §3. 시술 유형·특정 시술 텍스트 파싱 EDA\n\n"
       "**가설**: '특정 시술 유형'에 ICSI/IVF/IUI/BLASTOCYST/AH/FER/Unknown 등이 혼재"),
    code("# 3.1 시술 유형별 성공률\n"
         "print(train.groupby('시술 유형')[TARGET].agg(['count','mean']).round(4))\n"
         "print()\n"
         "# 특정 시술 유형 top 카테고리\n"
         "vc = train['특정 시술 유형'].value_counts().head(15)\n"
         "rate = train.groupby('특정 시술 유형')[TARGET].mean()\n"
         "top_df = pd.DataFrame({'count':vc, 'success_rate':rate.loc[vc.index]}).round(4)\n"
         "print('Top 15 특정 시술 유형:'); print(top_df)"),
    code("# 3.2 token 포함 여부 plug-and-play\n"
         "tokens = ['ICSI','IVF','IUI','BLASTOCYST','AH','FER','UNKNOWN','Unknown','알 수 없음']\n"
         "rows = []\n"
         "for t in tokens:\n"
         "    mask = train['특정 시술 유형'].fillna('').str.upper().str.contains(t.upper(), regex=False)\n"
         "    if mask.sum() > 0:\n"
         "        rows.append({'token':t, 'n':int(mask.sum()), 'lbr_with':train[mask][TARGET].mean(),\n"
         "                     'lbr_without':train[~mask][TARGET].mean()})\n"
         "tok_df = pd.DataFrame(rows)\n"
         "tok_df['delta'] = (tok_df['lbr_with']-tok_df['lbr_without']).round(4)\n"
         "print(tok_df.round(4))"),
    code("# 3.3 token 개수\n"
         "def count_tokens(s):\n"
         "    if pd.isna(s): return 0\n"
         "    return sum([1 for t in tokens if t.upper() in s.upper()])\n"
         "train['_token_count'] = train['특정 시술 유형'].apply(count_tokens)\n"
         "print(train.groupby('_token_count')[TARGET].agg(['count','mean']).round(4))"),
    md("**§3 후보 파생**: `treat_has_icsi`, `treat_has_ivf`, `treat_has_iui`, `treat_has_blastocyst`, `treat_has_ah`, `treat_has_fer`, `treat_has_unknown`, `treat_token_count`"),

    # =====================================================
    md("## §4. 배아·난자 수치 feature EDA\n\n"
       "**가설** (Zou 2025 RF AUC 0.808): retrieved oocyte / usable embryo / endometrial thickness가 핵심.\n"
       "본 데이터의 proxy: 총 생성 배아 수, 이식된 배아 수, 혼합된 난자 수, 미세주입된 난자 수"),
    code("num_cols = ['총 생성 배아 수','미세주입된 난자 수','미세주입에서 생성된 배아 수',\n"
         "            '이식된 배아 수','미세주입 배아 이식 수','저장된 배아 수','미세주입 후 저장된 배아 수',\n"
         "            '해동된 배아 수','해동 난자 수','수집된 신선 난자 수','저장된 신선 난자 수',\n"
         "            '혼합된 난자 수','파트너 정자와 혼합된 난자 수','기증자 정자와 혼합된 난자 수']\n"
         "rows = []\n"
         "for c in num_cols:\n"
         "    if c not in train: continue\n"
         "    s = train[c]\n"
         "    rows.append({'col':c, 'na%':round(s.isna().mean()*100,1),\n"
         "                 'zero%':round((s==0).mean()*100,1),\n"
         "                 'mean':round(s.mean(),2), 'median':round(s.median(),2),\n"
         "                 'p99':round(s.quantile(0.99),0), 'max':round(s.max(),0)})\n"
         "print(pd.DataFrame(rows).to_string(index=False))"),
    code("# 4.1 효율 ratio 계산 + target association\n"
         "ivf2 = train[train['시술 유형']=='IVF'].copy()\n"
         "ivf2['_생성_per_난자'] = ivf2['총 생성 배아 수'] / ivf2['혼합된 난자 수'].replace(0,np.nan)\n"
         "ivf2['_이식_per_생성'] = ivf2['이식된 배아 수'] / ivf2['총 생성 배아 수'].replace(0,np.nan)\n"
         "ivf2['_저장_per_생성'] = ivf2['저장된 배아 수'] / ivf2['총 생성 배아 수'].replace(0,np.nan)\n"
         "ivf2['_미세주입_효율'] = ivf2['미세주입에서 생성된 배아 수'] / ivf2['미세주입된 난자 수'].replace(0,np.nan)\n"
         "for c in ['_생성_per_난자','_이식_per_생성','_저장_per_생성','_미세주입_효율']:\n"
         "    s = ivf2[c].dropna()\n"
         "    print(f'{c}: median={s.median():.3f}, corr w/ target = {ivf2[c].corr(ivf2[TARGET]):.4f}')"),
    code("# 4.2 quantile bin별 성공률 (총 생성 배아 수)\n"
         "for c in ['총 생성 배아 수','이식된 배아 수','수집된 신선 난자 수']:\n"
         "    if c not in train: continue\n"
         "    bins = train[c].quantile([0,0.2,0.4,0.6,0.8,1.0]).unique()\n"
         "    if len(bins) >= 3:\n"
         "        train[c+'_bin'] = pd.cut(train[c], bins=bins, include_lowest=True, duplicates='drop')\n"
         "        print(f'{c}:')\n"
         "        print(train.groupby(c+'_bin', observed=True)[TARGET].agg(['count','mean']).round(4)); print()"),
    md("**§4 후보 파생**: `log1p_*` (모든 수치), `is_zero_*`, `is_missing_*`, `생성_per_난자`, `이식_per_생성`, `저장_per_생성`, `미세주입_효율`, `생성_minus_이식`, `생성_minus_이식저장`"),

    # =====================================================
    md("## §5. 배아 이식 전략 EDA\n\n"
       "**가설** (Sci Reports validated model): age × stage × fresh/frozen × N_transfer interaction이 핵심.\n"
       "live birth per embryo: 35세 fresh blastocyst 43% → 43세 frozen cleavage 1%"),
    code("# 5.1 이식된 배아 수별\n"
         "print('이식 배아 수별 성공률:')\n"
         "print(train.groupby('이식된 배아 수')[TARGET].agg(['count','mean']).round(4))\n"
         "print()\n"
         "# 5.2 단일 배아 이식 여부\n"
         "if '단일 배아 이식 여부' in train:\n"
         "    print('단일 배아 이식 여부별:')\n"
         "    print(train.groupby('단일 배아 이식 여부')[TARGET].agg(['count','mean']).round(4))"),
    code("# 5.3 배아 이식 경과일 day grouping\n"
         "def day_bin(d):\n"
         "    if pd.isna(d): return 'cancelled/NA'\n"
         "    if d <= 2: return 'day0-2'\n"
         "    if d == 3: return 'day3'\n"
         "    if d == 4: return 'day4'\n"
         "    if d == 5: return 'day5'\n"
         "    return 'day6+'\n"
         "train['_transfer_day_bin'] = train['배아 이식 경과일'].apply(day_bin)\n"
         "print(train.groupby('_transfer_day_bin')[TARGET].agg(['count','mean']).round(4))"),
    code("# 5.4 fresh / frozen / donor pattern\n"
         "for c in ['신선 배아 사용 여부','동결 배아 사용 여부','기증 배아 사용 여부']:\n"
         "    if c in train:\n"
         "        print(f'{c}:'); print(train.groupby(c)[TARGET].agg(['count','mean']).round(4)); print()\n"
         "# 5.5 fresh × frozen pattern\n"
         "if all(c in train.columns for c in ['신선 배아 사용 여부','동결 배아 사용 여부']):\n"
         "    train['_fresh_frozen_pattern'] = (train['신선 배아 사용 여부'].fillna(-1).astype(str) + '_' +\n"
         "                                     train['동결 배아 사용 여부'].fillna(-1).astype(str))\n"
         "    print(train.groupby('_fresh_frozen_pattern')[TARGET].agg(['count','mean']).round(4))"),
    md("**§5 후보 파생**: `has_embryo_transfer` (=배아 이식 경과일 not NA), `transfer_day_bin`, `transfer_day3`/`day5`, `fresh_frozen_pattern`, `age_x_transfer_bin`, `transfer_day_x_transfer_count`, `blastocyst_x_transfer_day` (token AND day=5)"),

    # =====================================================
    md("## §6. 불임 원인 EDA\n\n"
       "**가설**: 단일 원인보다 **원인 조합** 분석이 효과적. 남성/여성/부부/불명확 stratification."),
    code("cause_cols = [c for c in train.columns if '불임 원인' in c]\n"
         "print(f'불임 원인 컬럼 {len(cause_cols)}개')\n"
         "rows = []\n"
         "for c in cause_cols:\n"
         "    n_pos = (train[c]==1).sum()\n"
         "    if n_pos > 0:\n"
         "        lbr_y = train[train[c]==1][TARGET].mean()\n"
         "        lbr_n = train[train[c]!=1][TARGET].mean()\n"
         "        rows.append({'cause':c, 'n':int(n_pos), 'lbr_yes':round(lbr_y,4),\n"
         "                     'lbr_no':round(lbr_n,4), 'delta':round(lbr_y-lbr_n,4)})\n"
         "df_cause = pd.DataFrame(rows).sort_values('delta',key=abs, ascending=False)\n"
         "print(df_cause.head(20).to_string(index=False))"),
    code("# cause aggregates\n"
         "binary_cause = [c for c in cause_cols if (train[c].dropna().isin([0,1])).all() and (train[c]==1).sum()>0]\n"
         "train['_cause_total'] = train[binary_cause].sum(axis=1)\n"
         "print('원인 개수별 성공률:')\n"
         "print(train.groupby('_cause_total')[TARGET].agg(['count','mean']).round(4))"),
    code("# 남성/여성/부부 분리 카운트\n"
         "male_cause = [c for c in binary_cause if any(k in c for k in ['남성','정자'])]\n"
         "female_cause = [c for c in binary_cause if any(k in c for k in ['여성','난관','자궁','배란'])]\n"
         "couple_cause = [c for c in binary_cause if '부부' in c]\n"
         "train['_cause_male_count'] = train[male_cause].sum(axis=1)\n"
         "train['_cause_female_count'] = train[female_cause].sum(axis=1)\n"
         "train['_cause_couple_count'] = train[couple_cause].sum(axis=1)\n"
         "train['_cause_male_any'] = (train['_cause_male_count']>0).astype(int)\n"
         "train['_cause_female_any'] = (train['_cause_female_count']>0).astype(int)\n"
         "train['_cause_male_female_both'] = ((train['_cause_male_any']==1)&(train['_cause_female_any']==1)).astype(int)\n"
         "print(f'남성+여성 동시: n={train[\"_cause_male_female_both\"].sum()}, LBR={train[train[\"_cause_male_female_both\"]==1][TARGET].mean():.4f}')\n"
         "print(f'원인 없음: n={(train[\"_cause_total\"]==0).sum()}, LBR={train[train[\"_cause_total\"]==0][TARGET].mean():.4f}')"),
    md("**§6 후보 파생**: `cause_total`, `cause_any`, `cause_none`, `cause_multi`, `cause_male_count`, `cause_female_count`, `cause_male_any`, `cause_female_any`, `cause_male_female_both`, `unexplained_or_none`"),

    # =====================================================
    md("## §7. 난자/정자 출처 + donor age EDA"),
    code("# 7.1 난자 출처 × 정자 출처 cross\n"
         "ix = train.groupby(['난자 출처','정자 출처'])[TARGET].agg(['count','mean']).round(4)\n"
         "print(ix.sort_values('count', ascending=False).head(10))"),
    code("# 7.2 donor age (난자)\n"
         "donor = train[train['난자 출처']=='기증 제공']\n"
         "if len(donor) > 0:\n"
         "    print('기증 난자 사용 그룹 (n={}) 의 난자 기증자 나이별 LBR:'.format(len(donor)))\n"
         "    print(donor.groupby('난자 기증자 나이')[TARGET].agg(['count','mean']).round(4))\n"
         "# 7.3 donor age (정자)\n"
         "sperm_donor = train[train['정자 출처']=='기증 제공']\n"
         "if len(sperm_donor) > 0:\n"
         "    print('\\n기증 정자 사용 그룹의 정자 기증자 나이별 LBR:')\n"
         "    print(sperm_donor.groupby('정자 기증자 나이')[TARGET].agg(['count','mean']).round(4))"),
    md("**§7 후보 파생**: `egg_source`, `sperm_source`, `egg_sperm_source_cross`, `egg_donor_age_ord`, `sperm_donor_age_ord`, `egg_donor_age_unknown`, `sperm_donor_age_unknown`, age_x_egg_source"),

    # =====================================================
    md("## §8. 결측치 자체를 신호로\n\n"
       "**가설**: '해당 단계 미진행' / '해당 시술 미적용'을 의미할 수 있음"),
    code("rows = []\n"
         "for c in train.columns:\n"
         "    if c == TARGET: continue\n"
         "    n_na = train[c].isna().sum()\n"
         "    if 100 <= n_na < len(train)*0.99:\n"
         "        lbr_na = train.loc[train[c].isna(), TARGET].mean()\n"
         "        lbr_nn = train.loc[train[c].notna(), TARGET].mean()\n"
         "        delta = lbr_na - lbr_nn\n"
         "        if abs(delta) >= 0.01:\n"
         "            rows.append({'col':c, 'n_na':int(n_na),\n"
         "                         'lbr_na':round(float(lbr_na),4), 'lbr_nn':round(float(lbr_nn),4),\n"
         "                         'delta':round(float(delta),4)})\n"
         "df_na = pd.DataFrame(rows).sort_values('delta', key=abs, ascending=False)\n"
         "print(df_na.head(15).to_string(index=False))"),
    code("# 행 단위 missing count\n"
         "train['_row_missing_count'] = train.isna().sum(axis=1)\n"
         "bins = train['_row_missing_count'].quantile([0,0.25,0.5,0.75,0.9,1.0]).values\n"
         "train['_row_missing_bin'] = pd.cut(train['_row_missing_count'], bins=np.unique(bins), include_lowest=True)\n"
         "print(train.groupby('_row_missing_bin', observed=True)[TARGET].agg(['count','mean']).round(4))"),
    md("**§8 후보 파생**: `{col}_isna` (informative한 컬럼만), `row_num_missing_count`, `cancelled_cycle` (배아 이식 경과일 isna)"),

    # =====================================================
    md("## §9. Train-only Interaction Discovery"),
    code("# 우선순위 interaction 10개\n"
         "interactions = [\n"
         "    ('시술 당시 나이', '특정 시술 유형'),\n"
         "    ('시술 당시 나이', '이식된 배아 수'),\n"
         "    ('시술 당시 나이', '_transfer_day_bin'),\n"
         "    ('시술 당시 나이', '난자 출처'),\n"
         "    ('특정 시술 유형', '배아 생성 주요 이유'),\n"
         "    ('배아 생성 주요 이유', '이식된 배아 수'),\n"
         "    ('신선 배아 사용 여부', '동결 배아 사용 여부'),\n"
         "    ('난자 출처', '정자 출처'),\n"
         "    ('총 시술 횟수', '_prior_live_birth_any'),\n"
         "]\n"
         "for c1, c2 in interactions:\n"
         "    if c1 in train.columns and c2 in train.columns:\n"
         "        ix = train.groupby([c1,c2])[TARGET].agg(['count','mean']).round(4)\n"
         "        ix = ix[ix['count']>=100]\n"
         "        if len(ix) > 0:\n"
         "            print(f'\\n=== {c1} × {c2} (top 10 by count, n>=100) ===')\n"
         "            print(ix.sort_values('count', ascending=False).head(10))"),

    # =====================================================
    md("## §10. Leakage Audit\n\n"
       "**점검**: train→test의 모든 통계 산출 위치, OOF 적용, post-outcome 변수"),
    code("from src.validation.leakage_check import check_all\n"
         "result = check_all(str(DATA/'train.csv'), str(DATA/'test.csv'))\n"
         "import json; print(json.dumps(result, ensure_ascii=False, indent=2))"),
    code("# train과 test 컬럼 일치 확인\n"
         "tr_cols = set(train.columns) - {TARGET}\n"
         "te_cols = set(test.columns)\n"
         "print(f'only in train: {sorted(tr_cols - te_cols)}')\n"
         "print(f'only in test:  {sorted(te_cols - tr_cols)}')\n"
         "print('==> 둘 다 빈 list여야 함 (target만 train에 있음)')"),
    code("# Post-outcome 의심 변수 점검\n"
         "post_outcome_keywords = ['임신','출산','live birth','outcome','결과']\n"
         "suspect = [c for c in train.columns if any(k in c.lower() for k in [k.lower() for k in post_outcome_keywords]) and c != TARGET]\n"
         "print('Post-outcome 의심 컬럼:')\n"
         "for c in suspect:\n"
         "    print(f'  {c} (train mean={train[c].mean() if train[c].dtype != object else \"category\"})')\n"
         "print('\\n주의: 위 컬럼들은 cycle 이전(prior) 통계로 명확히 정의된 경우만 사용 가능')\n"
         "print('총 임신 횟수, 총 출산 횟수 = 본 cycle 이전 누적 → OK (사용 가능)')"),

    md("---\n## 종합 권장: §1~§10 발견을 반영한 학습 파이프라인\n\n"
       "1. 모든 횟수 변수: ordinal 정수 + censored flag\n"
       "2. 시술 유형 token 7개: `treat_has_*`\n"
       "3. 배아·난자 효율 ratio 4개\n"
       "4. fresh × frozen × donor pattern\n"
       "5. 불임 원인 집계 (총/남/여/부부)\n"
       "6. donor age ordinal\n"
       "7. informative NaN flag\n"
       "8. age × {시술 유형, transfer_day, transfer_count, donor} interaction\n\n"
       "→ `src/features/v2_features.py` 모듈로 통합 (다음 셀)"),
]

nb = {"cells":cells, "metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"}}, "nbformat":4, "nbformat_minor":5}
out = Path("/home/claude/v2_addon/notebooks/eda_master_v2.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"Created {out} ({len(cells)} cells)")
