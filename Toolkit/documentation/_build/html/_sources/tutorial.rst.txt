
#######################
Tutorial
#######################

.. contents:: 
    :depth: 4

	
Requirements
============

.. note:: Please find the requirements and the installion of Python, python libraries and CWatM in: :ref:`rst_setupdoc` 


Test the Python model version
=============================

**Windows and Linux** (and maybe Mac, but not tested)

Please try::


   python <modelpath>/run_cwatm.py  (for the Python3.+ version)
   or:
   <modelpath>/cwatm  (for the .exe version)

The output should be::

   Running under platform:  Windows  **(or Linux etc)** 
   CWatM - Community Water Model
   Authors: ...
   Version: ...
   Date: ...
   Arguments list:
   settings.ini     settings file
   -q --quiet       output progression given as .
   -v --veryquiet   no output progression is given
   -l --loud        output progression given as time st
   -c --check       input maps and stack maps are check
   -h --noheader    .tss file have no header and start
   -t --printtime   the computation time for hydrologic
   -w --warranty    copyright and warranty information   
	
.. warning:: If python is not set in the environment path, the full path of python has to be used



Running the model 1
===================


.. warning:: The model needs a settings file as an argument. See: :ref:`rst_settingdoc` 

python <modelpath>/cwatm.py settingsfile flags

example::

   python cwatm.py settings_rhine.ini -l
	
The flag -l show the output on screen as date and discharge 

At this point you should receive this eror message::

   ======================== CWatM FILE ERROR ===========================
   Cannot find option file: d:/work/CWatM/source/metaNetcdf.xml In  "metaNetcdfFile"
   searching: "d:/work/CWatM/source/metaNetcdf.xml"
   path: d:/work/CWatM/source does not exists	


Downloading and installing the spatial dataset 
==============================================

The spatial dataset contains:

* static data ie. data that does not change over time (a model assumption) e.g. soil data
* time dependend (inter annual) data that change periodical during a year e.g. crop coefficient of vegetation
* time dependend (intra annual) data that change by month or year e.g. fraction of landcover

These data are stored as global dataset:

* cwat_input.zip  for the 30' global version
* cwat_input5min.zip  for the 5' global version


As climate data different forcings can be used e.g:

* PGMFD v.2 (Princeton), GSWP3, etc.
* precipitation from e.g. MSWEP http://www.gloh2o.org/
* WATCH+WFDEI  https://www.isimip.org/gettingstarted/details/5/

and as projection e.g.:

* ISI-MIP dataset https://www.isimip.org/gettingstarted/#input-data-bias-correction




.. note:: 
   
    | Please copy and unpack the spatial dataset (either 30' or 5')in a folder
    | Please copy the the climate dataset 30min_meteo_rhine.zip or 5min_meteo_rhine.zip in a seperate folder
    | Please create a folder called output

.. note:: 
   
    | For testing purpose there is a file rhine_basin.zip on GitHub
    | it has all the necessary data to run the River Rhine on 30 arcmin from 1990-2010


Changing the Settings file
==========================
	
to run the model the pathes to data have to be set correctly:
The information of pathes are stored in the settings file around line 80-100

[FILE_PATHS]::

    PathRoot = E:/      
    PathOut = $(PathRoot)/output
    PathMaps = E:/cwatm_input
    PathMeteo = E:/climate
    #--------------------------------------
    [NETCDF_ATTRIBUTES]
    institution = IIASA
    title = Global Water Model - WATCH WDFEI
    metaNetcdfFile = $(FILE_PATHS:PathRoot)/CWatM/source/metaNetcdf.xml

.. note:: Please change the pathes according to your file system

.. _rst_output2:


Error and exception handling
============================

We try to make our program behave properly when encountering unexpected conditions. Therefore we caption a number of possible wrong inputs.

If you get an output with an error number please look at :ref:`rst_setuperror`




Changing parameters of the model
================================

.. note:: An overview of possibilities is given in  see :ref:`rst_settingdoc`






