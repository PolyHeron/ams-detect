# Air Mass Detector

Air Mass Detector is a machine learning pipeline that attempts to help meteorologists automatically identify when a substantial air mass change has occurred.

In metorological post-processing one might have a problem with quick substantial changes in air masses if a Kalman filter is used. Air Mass Detector is meant to be used to automatically identify such events, and at those prompt for a reset of the Kalman filter. The main of the project was carried out at [SMHI](https://www.smhi.se/en/about-smhi), with support from [UU](https://uu.se/en). The resulting report was:
[Using machine learning to identify the occurrence of changing air masses](http://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-357939).

## Prerequisites

Before you begin, ensure you have met the following requirements:
* You have installed Python version 3.6 or later (though even 2.7 should work with minor changes to the code).
* The installation instructions are for the `conda`/[`mamba`](https://github.com/mamba-org/mamba) package manager.
* It is probably very helpful to look at [the project report](http://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-357939) for navigating and using the code, although hopefully the commented code goes some way.

## Preparatory installations for running Air Mass Detector

To prepare for running Air Mass Detector, follow these steps in a conda command prompt (for Windows it might need to be elevated (admin)). [`mamba`](https://github.com/mamba-org/mamba) is used when possible, but `conda` can be used instead if preferred. `conda-forge` is the channel used here, but e.g. the `anaconda` channel would probably work as well:

Preferrably create a project specific environment first:
```
mamba create -n <environment_name> python=3.6 -c conda-forge
```

Activate the environment:
```
conda activate <environment_name>
```

Install the required packages:
```
mamba install matplotlib -c conda-forge
mamba install pandas -c conda-forge
mamba install scikit-learn -c conda-forge
mamba install seaborn -c conda-forge
mamba install pycairo -c conda-forge
mamba install datadotworld-py -c conda-forge
```

Create an account at [data.world](https://data.world/) (DDW), and get the data repository access key from
[Meteorological, Uppsala Automatic Weather Station, 1998-2017](https://data.world/aryoryte/meteorological-uppsala-automatic-weather-station-1998-2017).
There, in the drop down menu under `Explore this dataset` at the top right, choose `Open in third party app`. Then choose `Python`, `Get access code`, `Copy`.

Register/configure the DDW data repository access key by pasting it after running this command:
```
dw configure
```

If you do not already have a preferred editor/IDE installed, you can install Spyder:
```
mamba install spyder -c conda-forge
```

Run Spyder:
```
spyder
```
and open `ams-detect.py` in the editor.

If you are new to Spyder you might want to change some settings to your liking (see the built in tutorial/introdction).

## Using Air Mass Detector

Using Air Mass Detector is for now done through an IDE/editor (I have used Spyder), and one can run the whole code at once, but it is not yet adjusted for this (the resulting plots will be stacked on top of each other). A better, and at least somewhat convenient way of running it, is from the top, cell by cell (`Shift+`&#9166; (Windows)). Until you come to the plotting cell, where you might want to selectively run plot by desired plot (`Shift+F9` (Windows)).

## Contributing to Air Mass Detector
Although not very actively developed for the time being, if you want to contribute to Air Mass Detector, primarily follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -S -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively see the documentation some Git software forges have on collaboration and pull/patch/merge requests: <br />
CodeBerg: [Pull requests and Git flow](https://docs.codeberg.org/collaborating/pull-requests-and-git-flow/) <br />
SourceHut: [Contributing to projects on SourceHut](https://man.sr.ht/tutorials/#contributing-to-srht-projects) <br />
GitLab: [How to create a merge request](https://docs.gitlab.com/ce/user/project/merge_requests/creating_merge_requests.html) <br />
GitHub: [Creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## License

This project uses the following license: [AGPLv3](https://choosealicense.com/licenses/agpl-3.0/).
