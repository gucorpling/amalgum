import sys, math, os, io, re, copy
import numpy as np
from rfpimp import importances, plot_importances, plot_corr_heatmap, feature_corr_matrix
from collections import Counter, defaultdict
import scipy.stats as ss
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

script_dir = os.path.dirname(os.path.realpath(__file__))


def permutation_importances(clf, val_x, val_y, viz=False, log=True):
	out_dict = {}

	# Get feature importances via permutation
	if log:
		sys.stderr.write("o Feature permutation importances:\n\n")
	imp = importances(clf, val_x, val_y)
	for i in range(len(imp["Importance"])):
		key, val = imp.index[i], imp["Importance"].values[i]
		out_dict[key] = val
		if log:
			sys.stderr.write(key + "=" + str(val) + "\n")
	if viz:
		viz = plot_importances(imp)
		viz.view()

		viz = plot_corr_heatmap(val_x, figsize=(7,5))
		viz.view()
	if log:
		sys.stderr.write("\n")

	return out_dict


def report_correlations(data_x,thresh=0.9):
	no_zero_cols = data_x.loc[:, (data_x != 0).any(axis=0)]
	mat = feature_corr_matrix(no_zero_cols)
	mat.dropna(axis='columns')
	high_correlations = {}
	cols = list(mat.columns)
	for i, row in enumerate(mat.iterrows()):
		for j in cols[i+1:]:
			if not np.isnan(row[1].loc[j]):
				if row[1].loc[j] > thresh:
					high_correlations[(row[0],j)] = row[1].loc[j]
	return high_correlations


def report_theils_u(data,columns):
	thresh = 0.98
	implications = {}
	for col1 in columns:
		for col2 in columns:
			if col1 != col2:
				u = theils_u(data[col1],data[col2])
				if u > thresh:
					# Note if u is high, col2 predicts what col1 is
					implications[(col2,col1)] = u
	return implications


# Functions from https://github.com/shakedzy/dython/ (Apache 2.0)
def theils_u(x, y):
	s_xy = conditional_entropy(x,y)
	x_counter = Counter(x)
	total_occurrences = sum(x_counter.values())
	p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
	s_x = ss.entropy(p_x)
	if s_x == 0:
		return 1
	else:
		return (s_x - s_xy) / s_x
	

def conditional_entropy(x, y):
	"""
	Calculates the conditional entropy of x given y: S(x|y)
	Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy
	:param x: list / NumPy ndarray / Pandas Series
		A sequence of measurements
	:param y: list / NumPy ndarray / Pandas Series
		A sequence of measurements
	:return: float
	"""
	# entropy of x given y
	y_counter = Counter(y)
	xy_counter = Counter(list(zip(x,y)))
	total_occurrences = sum(y_counter.values())
	entropy = 0.0
	for xy in xy_counter.keys():
		p_xy = xy_counter[xy] / total_occurrences
		p_y = y_counter[xy[1]] / total_occurrences
		entropy += p_xy * math.log(p_y/p_xy)
	return entropy


def get_best_params(corpus, module_name, auto=""):

	subdir = "segmenters" if "seg" in module_name.lower() or "conn" in module_name.lower() else "sentencers"
	subdir += os.sep + "params"

	infile = script_dir + os.sep + subdir + os.sep + module_name + auto + "_best_params.tab"
	lines = io.open(infile).readlines()
	params = {}
	vars = []
	best_clf = None
	blank = False

	for line in lines:
		if "\t" in line:
			corp, clf_name, param, val = line.split("\t")
			val = val.strip()
			if corp == corpus:
				if blank:  # Corpus name seen for the first time since seeing blank line
					params = {}
					blank = False
				if param == "features":
					vars = val.split(",")
				else:
					if param == "best_score":
						continue
					if val.isdigit():
						val = int(val)
					elif re.match(r'[0-9]?\.[0-9]+$',val) is not None:
						val = float(val)
					elif val == "None":
						val = None
					params[param] = val
					best_clf = clf_name
		elif len(line.strip()) == 0:  # Blank line, prepare to reset params
			blank = True


	sys.stderr.write("o Restored best parameters for " + best_clf + "\n")

	if best_clf == "GradientBoostingClassifier":
		best_clf = GradientBoostingClassifier(random_state=42)
	elif best_clf == "ExtraTreesClassifier":
		best_clf = ExtraTreesClassifier(random_state=42)
	elif best_clf == "CatBoostClassifier":
		from catboost import CatBoostClassifier
		best_clf = CatBoostClassifier(random_state=42)
	elif best_clf == "XGBClassifier":
		from xgboost import XGBClassifier
		best_clf = XGBClassifier(random_state=42)
	elif best_clf == "RandomForestClassifier":
		best_clf = RandomForestClassifier(random_state=42)
	else:
		best_clf = None

	return best_clf, params, vars


def get_best_score(corpus, module_name):
	subdir = "segmenters" if "seg" in module_name.lower() else "sentencers"
	subdir += os.sep + "params"

	infile = script_dir + os.sep + subdir + os.sep + module_name + "_best_params.tab"
	lines = io.open(infile).readlines()
	best_score = 0.0

	for line in lines:
		if line.count("\t") == 3:
			corp, clf_name, param, val = line.split("\t")
			val = val.strip()
			if corp == corpus and param == "best_score":
				if float(val) > best_score:
					best_score = float(val)

	sys.stderr.write("o Found previous best score: " + str(best_score) + "\n")
	return best_score


def hyper_optimize(data_x,data_y,val_x=None,val_y=None,cat_labels=None,space=None,max_evals=20):
	from hyperopt import tpe, hp, space_eval
	from hyperopt.fmin import fmin
	from hyperopt.pyll.base import scope
	from sklearn.metrics import make_scorer, f1_score
	from sklearn.model_selection import cross_val_score, StratifiedKFold

	average="binary"
	if space is not None:
		if "average" in space:
			average = "micro"

	def f1_sklearn(truth,predictions):
		if space is not None:
			if "average" in space:
				return -f1_score(truth,predictions)
		return -f1_score(truth,predictions,average=average)

	f1_scorer = make_scorer(f1_sklearn)

	def objective(in_params):
		clf = in_params['clf']
		if clf == "rf":
			clf = RandomForestClassifier(n_jobs=4,random_state=42)
		elif clf == "gbm":
			clf = GradientBoostingClassifier(random_state=42)
		elif clf == "xgb":
			from xgboost import XGBClassifier
			clf = XGBClassifier(random_state=42,nthread=4)
		elif clf == "cat":
			from catboost import CatBoostClassifier
			clf = CatBoostClassifier(random_state=42)
		else:
			clf = ExtraTreesClassifier(n_jobs=4,random_state=42)

		if clf.__class__.__name__ == "XGBClassifier":
			params = {
				'n_estimators': int(in_params['n_estimators']),
				'max_depth': int(in_params['max_depth']),
				'eta': float(in_params['eta']),
				'gamma': float(in_params['gamma']),
				'colsample_bytree': float(in_params['colsample_bytree']),
				'subsample': float(in_params['subsample'])
			}
			if "average" in in_params:
				params["average"] = in_params["average"]
		else:
			params = {
				'n_estimators': int(in_params['n_estimators']),
				'max_depth': int(in_params['max_depth']),
				'min_samples_split': int(in_params['min_samples_split']),
				'min_samples_leaf': int(in_params['min_samples_leaf']),
				'max_features': in_params['max_features']
			}

		clf.set_params(**params)
		if val_x is None:  # No fixed validation set given, perform cross-valiation on train
			score = cross_val_score(clf, data_x, data_y, scoring=f1_scorer, cv=StratifiedKFold(n_splits=3), n_jobs=3).mean()
		else:  # validated on validation set
			clf.fit(data_x,data_y)
			pred_y = clf.predict(val_x)
			score = -f1_score(val_y,pred_y)
		if "Forest" in clf.__class__.__name__:
			shortname = "RF"
		elif "Cat" in clf.__class__.__name__:
			shortname = "CAT"
		elif "XG" in clf.__class__.__name__:
			shortname = "XGB"
		else:
			shortname = "ET" if "Extra" in clf.__class__.__name__ else "GBM"
		print("F1 {:.3f} params {} {}".format(-score, params, shortname))
		return score

	# For large corpora, consider raising max n_estimators up to 350
	if space is None:
		space = {
			'n_estimators': scope.int(hp.quniform('n_estimators', 75, 250, 10)),
			'max_depth': scope.int(hp.quniform('max_depth', 5, 40, 1)),
			'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
			'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 10, 1)),
			'max_features': hp.choice('max_features', ["sqrt", None, 0.5, 0.6, 0.7, 0.8]),
			'clf': hp.choice('clf', ["rf","et","gbm"])
		}

	sys.stderr.write("o Using "+str(data_x.shape[0])+" tokens to choose hyperparameters\n")
	if val_x is not None:
		sys.stderr.write("o Using "+str(val_x.shape[0])+" held out tokens as fixed validation data\n")
	else:
		sys.stderr.write("o No validation data provided, using cross-validation on train set to score\n")

	best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals)

	best_params = space_eval(space,best_params)
	sys.stderr.write(str(best_params) + "\n")

	best_clf = best_params['clf']
	if best_clf == "rf":
		best_clf = RandomForestClassifier(n_jobs=4,random_state=42)
	elif best_clf == "gbm":
		best_clf = GradientBoostingClassifier(random_state=42)
	elif best_clf == "xgb":
		from xgboost import XGBClassifier
		best_clf = XGBClassifier(random_state=42,nthread=4)
	else:
		best_clf = ExtraTreesClassifier(n_jobs=4,random_state=42)
	del best_params['clf']
	best_clf.set_params(**best_params)

	return best_clf, best_params


def grid_search(data_x,data_y,tune_mode,clf_params):
	best_params = {}
	best_params_by_clf = defaultdict(dict)
	# Tune individual params separately for speed, or do complete grid search if building final model
	params_list = [{"n_estimators":[125,150,175,200,225]},
				   {'max_depth': [15,30,None]},
				   {"min_samples_split": [2, 5, 10]},
				   {"min_samples_leaf":[1,2,3]},
				   {"max_features":[None,"sqrt","log2"]}]
	if tune_mode == "full":
		# Flatten dictionary if doing full CV
		params_list = [{k: v for d in params_list for k, v in d.items()}]
	best_score = -10000
	for clf in [RandomForestClassifier(),ExtraTreesClassifier(),GradientBoostingClassifier()]:
		for params in copy.deepcopy(params_list):
			if "max_features" in params:
				if clf.__class__.__name__ == "GradientBoostingClassifiwr":
					params["max_features"] = [None,4,"sqrt"]
				elif clf.__class__.__name__ == "ExtraTreesClassifier":
					params["max_features"] = [None]
			base_params = copy.deepcopy(clf_params)  # Copy default params
			if clf.__class__.__name__ != "GradientBoostingClassifier":
				base_params.update({"n_jobs":4, "oob_score":True, "bootstrap":True})
			else:
				for p in ["n_jobs", "oob_score", "bootstrap"]:
					if p in base_params:
						base_params.pop(p)
			for p in params:
				if p in base_params:  # Ensure base_params don't conflict with grid search params
					base_params.pop(p)
			clf.set_params(**base_params)
			grid = GridSearchCV(clf,params,cv=3,n_jobs=3,error_score="raise",refit=False,scoring="neg_log_loss")  # We want accurate probabilities, so score neg_log_loss
			grid.fit(data_x,data_y)
			if tune_mode == "full":
				if grid.best_score_ > best_score:
					best_score = grid.best_score_
					best_clf = clf
					for param in params:
						best_params[param] = grid.best_params_[param]
			else:
				if grid.best_score_ > best_score:
					best_clf = clf
				for param in params:
					best_params_by_clf[clf.__class__.__name__][param] = grid.best_params_[param]
	if tune_mode == "paramwise":
		best_params = best_params_by_clf[best_clf.__class__.__name__]
	else:
		best_params["best_score"] = best_score

	return best_clf, best_params
