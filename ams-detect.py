# -*- coding: utf-8 -*-

# %%


"""
READ DATA
"""

# %%  Imports.

import datadotworld as dw
import io
import pandas as pd

# %%  Reading, parsing, naming, upsampling and interpolating the data.

#  Loading the data (CSV) from: https://data.world/aryoryte/
DDWUsrDir = "aryoryte/"
DDWUaDir = "meteorological-uppsala-automatic-weather-station-1998-2017"
DDWUaDir = DDWUsrDir+DDWUaDir
dataPath = "original/Uppsala 1998 till 2017.csv"
UaCSV = dw.load_dataset(DDWUaDir).raw_data[dataPath]

#  Names given to the input variuables/features/...
colnames = ['UTC', 'windDir', 'windSpeed', 'airTemp', 'dewPt', 'relHum']

#  Reading the data from the CSV file into a Pandas dataframe; headers
#  are at row 6 (but given the names colnames), trying to skip the last
#  row which is blank/empty but an initial space - fails..., choosing
#  parse_dates True to most quickly (C engine) parse the dates of the
#  UTC/index column.
df = pd.read_csv(io.StringIO(UaCSV.decode('utf-8')),
                 header=6, sep=';', names=colnames,
                 index_col='UTC', skipinitialspace=True,
                 skiprows=range(5, 10), skip_blank_lines=True,
                 parse_dates=True, infer_datetime_format=True)

#  Dropping last "row" which failed with skipinitialspace &
#  skip_blank_lines.
df = df[:-1]

#  Upsampling to include all hours, and interpolates over the
#  NaNs/NULLs/NaTs.
dfui = df.asfreq('H').interpolate()  # u(psample)i(nterpolate).

# %%


"""
SIGNAL PROCESSING - ALGORITHMICALLY LABEL DATA
"""

# %%  Imports.

import numpy as np
import datetime as dt

# %%  Declaration of constants.

#  Threshold/delta (degree Celsius) that defines the detecion of AMS:s.
thresh = 2.5  # Tried: 3, 3.5, 2.5

#  Size (days before/after detection) of true and false windows.
dayBfr = 1
dayAft = 3
#  Thoughts about up to 6. But maybe needs to be less than 3 for
#  forecast data.

#  Seed for generating pseudorandoms.
rndSeed = 5555

# %%  Aggregating and rolling statistics.

#  Diurnal air temperature mean (arithmetic (henceforth)).
dfui_m = dfui.airTemp.resample('D').mean()

#  Two-day air temperature mean, daily stride.
#  2rm2:
#  #WINDOW#r(-olling)av(era)g(e)2  (m(m)=m^2=m2)
dfui_2rm2 = dfui_m.rolling(window='2D', center=False,
                                closed='right').mean()
#  One could reflect on the kind of interval the mean is/(should be)
#  taken over, for now one closed on the right.

# %%  Original true detections.

#  Each minus that two days fwd, as a two-day window is used.
dfui_2rm2_dff = dfui_2rm2.diff(-2)

dfui_2rm2_dff_evn = dfui_2rm2_dff.iloc[::2]
dfui_2rm2_dff_odd = dfui_2rm2_dff.iloc[1::2]

#  True where one is larger than the other, and larger than the
#  threshold.
dfui_2rm2_dff_evn_msk = ((abs(dfui_2rm2_dff_evn)
                            > abs(dfui_2rm2_dff_odd.reindex(index=
                            dfui_2rm2_dff_evn.index, method='bfill')))
                            & (abs(dfui_2rm2_dff_evn) > thresh))
dfui_2rm2_dff_odd_msk = ((abs(dfui_2rm2_dff_odd)
                            >= abs(dfui_2rm2_dff_evn.reindex(index=
                            dfui_2rm2_dff_odd.index, method='ffill')))
                            & (abs(dfui_2rm2_dff_odd) > thresh))

#  Only Trues - for the indicator/flag plotting.
dfui_2rm2_dff_evn_tru = dfui_2rm2_dff_evn_msk[dfui_2rm2_dff_evn_msk]
dfui_2rm2_dff_odd_tru = dfui_2rm2_dff_odd_msk[dfui_2rm2_dff_odd_msk]

#  For plotting.
dfui_2rm2_dff_tru = pd.concat([dfui_2rm2_dff_evn_tru,
                                 dfui_2rm2_dff_odd_tru]).sort_index()

# %%  True event detections aggregated from the originals.

#  agg(regate)M(a)sk - collapsing those one, then two, then three days
#  after each other.
aggMsk_1 = (dfui_2rm2_dff_tru & ~(dfui_2rm2_dff_tru
            & dfui_2rm2_dff_tru.shift(1, freq='D'))).reindex(index= \
            dfui_2rm2.index, fill_value=False)
aggMsk_2 = (dfui_2rm2_dff_tru & ~(dfui_2rm2_dff_tru
            & dfui_2rm2_dff_tru.shift(2, freq='D'))).reindex(index= \
            dfui_2rm2.index, fill_value=False)
aggMsk_3 = (dfui_2rm2_dff_tru & ~(dfui_2rm2_dff_tru
            & dfui_2rm2_dff_tru.shift(3, freq='D'))).reindex(index= \
            dfui_2rm2.index, fill_value=False)
aggMsk = aggMsk_1 & aggMsk_2 & aggMsk_3
aggTru = aggMsk[aggMsk]
aggTru.rename('event', inplace=True)

# %%  Extraction of windows from events detected.

#  i(nter)v(als)
aggTru_iv = pd.DataFrame(data=0, index=aggTru.index, columns=['L', 'U'])
aggTru_iv.L = aggTru.shift(-dayBfr, freq='D').index  # Lower bound
aggTru_iv.U = aggTru.shift(dayAft, freq='D').index  # Upper bound

#  For plotting.
dfui_wins = [dfui.airTemp[(dfui.index >= aggTru_iv.L[i])
             & (dfui.index < aggTru_iv.U[i])]
             for i in range(len(aggTru_iv))]
dfui_wins_concat = pd.concat(dfui_wins)

#  For feeding to logistic regression.
dfui_wins_intidxlst = [dfui.airTemp[(dfui.index >= aggTru_iv.L[i])
                       & (dfui.index < aggTru_iv.U[i])].reset_index(drop=True)
                       for i in range(len(aggTru_iv))]
dfui_wins_intidxdf = pd.DataFrame(data=dfui_wins_intidxlst)
dfui_wins_intidxdf.reset_index(drop=True, inplace=True)
dfui_wins_intidxdf['event'] = True

# %%  Generation of false events.

np.random.seed(seed=rndSeed)
falseTimes = np.random.choice(dfui.index, size=len(dfui_wins))

# %%  Extraction of windows from false events generated.

#  i(nter)v(als)
falseTimes_iv = pd.DataFrame(data=0, index=falseTimes, columns=['L', 'U'])
falseTimes_iv.L = falseTimes_iv.shift(-dayBfr, freq='D').index  # Lower bound
falseTimes_iv.U = falseTimes_iv.shift(dayAft, freq='D').index  # Upper bound

#  For plotting.
dfui_falseWins = [dfui.airTemp[(dfui.index >= falseTimes_iv.L[i])
                  & (dfui.index < falseTimes_iv.U[i])]
                  for i in range(len(falseTimes_iv))]
dfui_falseWins_concat = pd.concat(dfui_falseWins)

#  For feeding to logistic regression.
dfui_falseWins_intidxlst = [dfui.airTemp[(dfui.index >= falseTimes_iv.L[i])
                            & (dfui.index < falseTimes_iv.U[i])].reset_index(
                            drop=True) for i in range(len(falseTimes_iv))]
dfui_falseWins_intidxdf = pd.DataFrame(data=dfui_falseWins_intidxlst)
dfui_falseWins_intidxdf.reset_index(drop=True, inplace=True)
dfui_falseWins_intidxdf['event'] = False

# %%  Preparation for feeding to logistic regression.

#  Formatted for training - with RangeIndex.
dfui_training_concat = pd.concat([dfui_wins_intidxdf,
                                  dfui_falseWins_intidxdf]).reset_index(
                                  drop=True)

#  Analogous to training formatting but with DatetimeIndex.
training_times = pd.concat([aggTru, pd.Series(data=False, index=falseTimes,
                                              name='event')])
#  falseTimes was datetime64[ns] np ndarray. Serialized to be
#  concatenated.

dfui_wins_concat_axis1 = pd.concat(dfui_wins, axis=1)
dfui_falseWins_concat_axis1 = pd.concat(dfui_falseWins, axis=1)
dfui_concat_axis1 = pd.concat([dfui_wins_concat_axis1,
                               dfui_falseWins_concat_axis1], axis=1)
dfui_concat_axis1.columns = range(0, len(dfui_concat_axis1.columns))

# %%


"""
SIGNAL PROCESSING - LOGISTICALLY REGRESS LABELLED DATA
"""

# %%  Imports.

from sklearn import linear_model
from sklearn.model_selection import train_test_split

# %%  Constants.

#  Seed for generating pseudorandoms.
rndState = 42

# %%  Training, fitting and testing.

#  Simple train/test-split.
X_train, X_test, y_train, y_test = train_test_split(
        dfui_training_concat.loc[:, dfui_training_concat.columns != 'event'],
        dfui_training_concat.loc[:, dfui_training_concat.columns == 'event'],
        test_size=0.33, random_state=rndState)

#  Defining the linear model.
#  C = "Penalty weight (often named lambda)." (default C=1.0)
#  penalty = "Norm applied on the model parameters (betas), here the
#  L1-norm (LASSO) is the choice.
#  tol = "Tolerance for the stopping criteria of the nonlinear solver."
clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-4,
                                      max_iter=1000)
# The L1-norm (LASSO) was original choice, but eventually changed to
# the L2-norm due to incompatibility with newer scikit-learn after
# switching to Python 3."
# max_iter = The maximum number of iterations to ..., needed an increase
# to something larger than the (new? (Py3)) 100 default.
# solver = The saga solver could be an alternative to the above...

clf.fit(X_train, np.ravel(y_train))

X_pred_bool = clf.predict(X_test)

X_pred = pd.DataFrame(data=X_pred_bool, index=X_test.index, columns=['event'])

prcnt = np.sum(X_pred == y_test)/len(X_test)
print("Logistic regression detection rate:  ",
      (round(prcnt, 3)*100).to_string(index=False), "%")
# str(round(prcnt, 3)*100) did not work as intended anymore under Python 3.

# %%  Preparing for plotting.

#  For showing which classified true/false in testing.
X_pred_tru = X_pred[X_pred.event]
X_pred_false = X_pred[~X_pred.event]
dfui_pred = pd.DataFrame(
                data=np.nan, index=range(0, len(dfui_training_concat)),
                columns=['event']).join(X_pred_tru, how='left',
                lsuffix="_left", rsuffix="_center")
dfui_pred = dfui_pred.event_center
dfui_pred = pd.DataFrame(
                data=dfui_pred.values,
                index=range(0, len(dfui_training_concat)),
                columns=['event']).join(X_pred_false, how="left",
                lsuffix="_center", rsuffix="_right")
dfui_pred.event_center.fillna(value=-1, inplace=True)
nyan = pd.DataFrame(data=-1, index=range(0, len(dfui_training_concat)),
                    columns=['nyan'])
dfui_pred.event_center[dfui_pred.event_center == nyan.nyan] =\
    dfui_pred.event_right[dfui_pred.event_center == nyan.nyan]
dfui_pred = dfui_pred.event_center
#  Corresponding plotting routine commented out for now.
#dfui_concat_axis1_tru = dfui_concat_axis1.loc[:, dfui_pred == True]
#  Corresponding plotting routine commented out for now.
#dfui_concat_axis1_false = dfui_concat_axis1.loc[:, dfui_pred == False]
#  Has no corresponding plotting routine for now.
#testing_classified = training_times.loc[
#    ((dfui_pred == True) | (dfui_pred == False)).values]
testing_classdTru = training_times[(dfui_pred == True).values]
testing_classdFalse = training_times[(dfui_pred == False).values]

#  Creating timeseries to show the occurrences and the non-occurrences
#  tested on.
testing_times = pd.concat([pd.Series(data=True, index=X_test.index),
                           pd.Series(data=False, index=X_train.index)])
testing_times.sort_index(inplace=True)
training_tested = training_times[testing_times.values]
testing_TruLabeld = training_tested[training_tested]
testing_FalseLabeld = training_tested[~training_tested]

# %%


"""
PLOT DATA
"""

# %%  Imports.

#  Switching backend.
import matplotlib
matplotlib.use('Qt5Cairo')  # Cairo vector renders the Qt5 canvas.

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.transforms as tx

from collections import OrderedDict

import IPython

# %%

#  Switching from Spyder default plotting ("%matplotlib inline") to
#  "%matplotlib qt".
shell = IPython.get_ipython()
shell.enable_matplotlib()
#  This defaults to spec in matplotlibrc (qt5agg).
#  Alt spec (gui='qt' or) 'inline'.

# %%  Seaborn plot settings.

sns.set_style('whitegrid')
sns.set_style('ticks', {"grid.linestyle": "--"})
sns.set_context('poster')
# notebook (def), paper, talk, poster  # font_scale=1.23

# %%  Declaring a dictionary and some tuples.

vdFont = {"fontname": "Verdana"}
color = (0, 0, 0, 1)  # Just Black
xLabelCoords = (1.042, 0.0097)  # 'poster'  # 'paper': 1.02, 0.006
yLabelCoords = (0, 1.02)
minMaxTemp = (-53, 39)  # i.e. 38 (because range() open to the right)

# %%  Defining the date format.

datesFmt = mdates.DateFormatter('%Y-%m-%d')

# %%  Plotting (k(ey)w(ord)arg(ument)s figure, axes and linestyle may
#     sometimes be redundant.)

#  Creating a figure object and an axes object.
fig, ax = plt.subplots()

#  Setting up x and y labels.
ax.set_xlabel('Time', color='sienna', **vdFont)
ax.xaxis.set_label_coords(*xLabelCoords)
ax.set_ylabel(u'Air temperature (\u00B0C)', color='navy',
               rotation='horizontal', **vdFont)
ax.yaxis.set_label_coords(*yLabelCoords)

#  Plotting disks (circles) for the original data points.
ax.plot(df.index, df.airTemp, color='navy', marker='o', linestyle='None',
        label="Observations")

#  Plotting lines between the hourly datapoints obtained from
#  upsampling and interpolating.
ax.plot(dfui.index, dfui.airTemp, color='navy', linestyle='-', alpha=0.67,
         label="Observations upsampled and interpolated")

#  Plotting disks and stepwise plateaus indicating the diurnal mean.
#  The mean value calculated for one day is kept throughout the day
#  and changed "post", i.e. at the next diurnal mean value.
ax.step(dfui_2rm2.index, dfui_2rm2, color='maroon',
         where='post', marker='o', alpha=0.8,
         label="Two-day windowed diurnal stride mean")

#  Transformation such that x-coordinates will be relative to the data,
#  and y-coordinates will be relative to the axes, i.e. the plotting
#  area. (x counts in time, y counts in the real interval [0, 1])
trans = tx.blended_transform_factory(ax.transData, ax.transAxes)

#  Plotting two-day mean delta occurrence detections.
#  k(ey)w(ord)arg(ument)s
kwargs_dfui_2rm2_dff_tru = dict(linewidth=2, color='seagreen', transform=trans,
                                label=
                                "Two-day mean delta occurrence detections")
plt.plot(np.repeat(dfui_2rm2_dff_tru.index, 3), np.tile([.05, .95, np.nan],
         len(dfui_2rm2_dff_tru)), **kwargs_dfui_2rm2_dff_tru)

#  Plotting aggregated/updated occurrence detections.
kwargs_aggTru = dict(alpha=0.33, axes=ax, figure=fig, linewidth=10,
                     color='fuchsia', transform=trans,
                     label="Aggregated/updated occurrence detections")
plt.plot(np.repeat(aggTru.index, 3), np.tile([.1, .9, np.nan], len(aggTru)),
         **kwargs_aggTru)

#  Plotting windows extracted w.r.t. updated occurrence detections).
kwargs_dfui_wins_concat = dict(
        alpha=0.15, antialiased=True, axes=ax, figure=fig, marker='o',
        markeredgecolor='grey', markeredgewidth=2, markerfacecolor='fuchsia',
        markersize=20, linestyle='None',
        label="Windows extracted w.r.t. updated occurrence detections")
ax.plot(dfui_wins_concat.index, dfui_wins_concat,
         **kwargs_dfui_wins_concat)

#  Plotting randomly generated non-occurrences.
kwargs_falseTimes = dict(alpha=0.33, axes=ax, figure=fig, linewidth=15,
                         color='gold', transform=trans,
                         label="Randomly generated non-occurrences")
plt.plot(np.repeat(falseTimes, 3), np.tile([.125, .875, np.nan],
         len(falseTimes)), **kwargs_falseTimes)

#  Plotting windows extracted w.r.t. non-occurrences.
kwargs_dfui_falseWins_concat = dict(
        alpha=0.15, antialiased=True, axes=ax, figure=fig, marker='o',
        markeredgecolor='grey', markeredgewidth=2, markerfacecolor='gold',
        markersize=20, linestyle='None',
        label="Windows extracted w.r.t. non-occurrences")
ax.plot(dfui_falseWins_concat.index, dfui_falseWins_concat,
         **kwargs_dfui_falseWins_concat)

#  Here one could have a plotting routine for showing all the times
#  tested on - on equal footing, in one step.
#  But for now showing them distinctly, in two separate steps:

#  Plotting detected occurrences tested on.
kwargs_testing_TruLabeld = dict(alpha=0.33, axes=ax, figure=fig, linewidth=20,
                                color='orangered', transform=trans,
                                label="Detected occurrences tested on")
plt.plot(np.repeat(testing_TruLabeld.index, 3), np.tile([.25, .75, np.nan],
         len(testing_TruLabeld)), **kwargs_testing_TruLabeld)

#  Plotting non-ocurrences tested on.
kwargs_testing_FalseLabeld = dict(alpha=0.33, axes=ax, figure=fig,
                                  linewidth=25, color='lime',
                                  transform=trans,
                                  label="Non-occurrences tested on")
plt.plot(np.repeat(testing_FalseLabeld.index, 3), np.tile([.33, .67, np.nan],
         len(testing_FalseLabeld)), **kwargs_testing_FalseLabeld)

#  Plotting true classifications.
kwargs_testing_classdTru = dict(alpha=0.33, axes=ax, figure=fig, linewidth=30,
                                color='darkorchid', transform=trans,
                                label="Classified True")
plt.plot(np.repeat(testing_classdTru.index, 3), np.tile([.4, .6, np.nan],
         len(testing_classdTru)), **kwargs_testing_classdTru)

##  Plotting windows associated with true classifications.
#kwargs_dfui_concat_axis1_tru = dict(
#        alpha=0.15, antialiased=True, axes=ax, figure=fig, marker='o',
#        markeredgecolor='grey', markeredgewidth=2,
#        markerfacecolor='darkorchid', markersize=20, linestyle='None',
#        label="Windows associated with classified true")
#ax.plot(dfui_concat_axis1_tru.index, dfui_concat_axis1_tru,
#         **kwargs_dfui_concat_axis1_tru)
#  Calculation/creation (log_reg_data) commented out for now.

#  Plotting false classifications.
kwargs_testing_classdFalse = dict(alpha=0.66, axes=ax, figure=fig,
                                  linewidth=35, color='orange',
                                  transform=trans, label="Classified False")
plt.plot(np.repeat(testing_classdFalse.index, 3), np.tile([.45, .55, np.nan],
         len(testing_classdFalse)), **kwargs_testing_classdFalse)

##  Plotting windows associated with false classifications.
#kwargs_dfui_concat_axis1_false = dict(
#        alpha=0.15, antialiased=True, axes=ax, figure=fig, marker='o',
#        markeredgecolor='grey', markeredgewidth=2,
#        markerfacecolor='orange', markersize=20, linestyle='None',
#        label="Windows associated with classified false")
#ax.plot(dfui_concat_axis1_false.index, dfui_concat_axis1_false,
#         **kwargs_dfui_concat_axis1_false)
#  Calculation/creation (log_reg_data) commented out for now.

#  Setting up the y ticks.
ax.tick_params(axis='y', labelcolor=color)
ax.set_yticks(list(range(*minMaxTemp)), minor=True)

#  Setting the date format.
ax.xaxis.set_major_formatter(datesFmt)

#  Setting up the grid. (Should partly be set by the Seaborn call.)
#kwargs_majorXGrid = dict(alpha=.42, markevery='None', xdata=dfui_m.index)
ax.grid(b=True, which='major', axis='x', alpha=.42)  # **kwargs_majorXGrid ...
#  ... xdata=dfui_m.index acceptance ceased ...
#  ... with update matplotlib 2.1.2 -> 2.2.2
ax.grid(b=True, which='major', axis='y', alpha=.42)
ax.grid(b=True, which='minor', axis='y', alpha=.21)

#  Formatting the legend.
handles, labels = plt.gca().get_legend_handles_labels()
handles[0].set_linestyle('None')
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best',
           labelspacing=1.8, handletextpad=1.8)
# loc=(.015, .52)  # loc='best'  # loc=(.013, .59)  # loc=(.013, .67)

#  Removing the upper and right spines.
sns.despine()

#  Not needed for now.
# plt.xticks(rotation='vertical')

#  Not needed for now.
# plt.show()

# %%
