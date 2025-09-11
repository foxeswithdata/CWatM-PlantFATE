# -*- coding: utf-8 -*-
"""
CWatM Variable Documentation Generator

This program automatically scans CWatM (Community Water Model) Python modules to extract,
document, and maintain variable definitions used throughout the hydrological modeling system.

Main Functionality:
- Scans Python files in specified folders to find all 'self.var' variable references
- Tracks variable definitions vs. usage across different modules  
- Creates comprehensive Excel documentation with variable descriptions, units, and priorities
- Generates NetCDF metadata XML files for model output variables
- Automatically injects variable documentation tables into Python module docstrings

The program helps maintain consistent variable documentation across the entire CWatM codebase
by providing a centralized system for tracking and documenting model variables. It supports
interactive editing workflows where users can manually update variable descriptions and
metadata through Excel files.

Output Files:
- Excel workbooks with variable documentation (default: selfvar.xlsx)
- NetCDF metadata XML files (default: metaNetcdf.xml)
- Updated Python module files with embedded variable documentation tables

Created on Tue Apr  7 15:13:10 2020
@author: Luca G., Peter B.
"""
from builtins import isinstance

'''
Requires:
openpyxl
loguru

'''

import argparse
import json
import os
import platform
import re
import time
from pathlib import Path
# PB added to sort the Dict_AllVariables
from collections import OrderedDict
from operator import getitem
from xml.dom import minidom
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set

import numpy as np
import pandas as pd
from loguru import logger


base_folders = [
    'hydrological_modules',
    'hydrological_modules/routing_reservoirs',
    'hydrological_modules/groundwater_modflow',
    'hydrological_modules/water_demand',
    'management_modules',
    '.'
]

xcols = [
    'Variable name',
    'Long name',
    'Unit',
    'Description',
    'Type',
    'First module',
    'Priority',
    'Module 1', 'Module 2', 'Module 3', 'Module 4',
    'Module 5', 'Module 6', 'Module 7', 'Module 8',
    'Module 9', 'Module 10', 'Module 11', 'Module 12',
    'Module 13', 'Module 14', 'Module 15', 'Module 16'
]


netxml_head = (
    "<CWATM>\n" +
    "# METADATA for NETCDF OUTPUT DATA\n\n" +
    "# varname: name of the variable in the CWAT code\n" +
    "# unit: unit of the varibale\n" +
    "# long name# standard name\n\n" +
    '# Time information\n' +
    '<metanetcdf varname="_daily"     time=": daily"/>\n' +
    '<metanetcdf varname="_monthavg"  time=": monthly average"/>\n' +
    '<metanetcdf varname="_monthend"  time=": last value of the month"/>\n' +
    '<metanetcdf varname="_monthtot"  time=": monthly sum"/>\n' +
    '<metanetcdf varname="_annualavg" time=": annual average"/>\n' +
    '<metanetcdf varname="_annualend" time=": last value of the year"/>\n' +
    '<metanetcdf varname="_annualtot" time=": annual sum"/>\n' +
    '<metanetcdf varname="_totalavg"  time=": average over the whole time period"/>\n' +
    '<metanetcdf varname="_totaltot"  time=": sum the whole time period"/>\n' +
    '<metanetcdf varname="_totalend"  time=": last value of the whole time period"/>\n\n\n'
)


@dataclass
class Config:
    """Enhanced configuration for the variable documentation generator."""
    # Core functionality (original)
    excel_file: Optional[str] = None
    xml_file: Optional[str] = None
    folders: Optional[List[str]] = None
    auto_open_excel: bool = True
    inject_docs: bool = True
    
    # Optional enhancements
    config_file: Optional[str] = None
    batch_mode: bool = False
    export_formats: List[str] = field(default_factory=list)  # csv, markdown, html
    filter_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    validate_consistency: bool = False
    use_cache: bool = False
    cache_file: str = '.variable_cache.json'
    log_level: str = 'INFO'
    template_dir: Optional[str] = None
    output_dir: str = '.'
    project_name: Optional[str] = None
    
    def get_excel_file(self) -> str:
        """Get Excel filename with default fallback."""
        base_name = self.excel_file or 'selfvar'
        if self.project_name:
            base_name = f"{self.project_name}_{base_name}"
        return f"{base_name}.xlsx"
    
    def get_xml_file(self) -> str:
        """Get XML filename with default fallback."""
        base_name = self.xml_file or 'metaNetcdf'
        if self.project_name:
            base_name = f"{self.project_name}_{base_name}"
        return f"{base_name}.xml"
    
    def get_folders(self) -> List[str]:
        """Get folder list with default fallback."""
        return self.folders or ['../' + f for f in base_folders]
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)


class VariableFilter:
    """Optional variable filtering and validation."""
    
    def __init__(self, config: Config):
        self.include_patterns = [re.compile(p) for p in config.filter_patterns]
        self.exclude_patterns = [re.compile(p) for p in config.exclude_patterns]
    
    def should_include(self, var_name: str) -> bool:
        """Check if variable should be included based on filters."""
        # If no include patterns specified, include all
        if not self.include_patterns:
            include = True
        else:
            include = any(pattern.search(var_name) for pattern in self.include_patterns)
        
        # Check exclude patterns
        exclude = any(pattern.search(var_name) for pattern in self.exclude_patterns)
        
        return include and not exclude


class CacheManager:
    """Optional caching for large codebases."""
    
    def __init__(self, cache_file: str, use_cache: bool = False):
        self.cache_file = cache_file
        self.use_cache = use_cache
        self._cache = {}
        self._file_mtimes = {}
        
        if use_cache and os.path.exists(cache_file):
            self._load_cache()
    
    def _load_cache(self):
        """Load cache from file."""
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
                self._cache = data.get('variables', {})
                self._file_mtimes = data.get('mtimes', {})
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")
    
    def _save_cache(self):
        """Save cache to file."""
        if not self.use_cache:
            return
        try:
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'variables': self._cache,
                    'mtimes': self._file_mtimes,
                    'timestamp': time.time()
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
    
    def get_variables(self, file_path: str) -> Optional[Dict]:
        """Get cached variables if file hasn't changed."""
        if not self.use_cache:
            return None
        
        try:
            current_mtime = os.path.getmtime(file_path)
            cached_mtime = self._file_mtimes.get(file_path)
            
            if cached_mtime == current_mtime and file_path in self._cache:
                return self._cache[file_path]
        except OSError:
            pass
        
        return None
    
    def set_variables(self, file_path: str, variables: Dict):
        """Cache variables for a file."""
        if not self.use_cache:
            return
        
        try:
            self._cache[file_path] = variables
            self._file_mtimes[file_path] = os.path.getmtime(file_path)
            self._save_cache()
        except OSError as e:
            logger.warning(f"Could not cache variables for {file_path}: {e}")


def setup_logging(level: str = 'INFO'):
    """Configure logging with specified level."""
    logger.remove()  # Remove default logger
    log_format = ("<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | " +
                  "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    
    logger.add(
        lambda msg: print(msg, end=''),
        format=log_format,
        level=level,
        colorize=True
    )


def open_workbook(wbook_file):
    """
    Open the workbook file using the system's default spreadsheet application.
    
    Opens the argument workbook file using Excel under Windows operating system
    or LibreOffice Calc under Linux. MacOS is not currently supported.
    
    Parameters
    ----------
    wbook_file : str
        Path to the workbook file to be opened.
    
    Returns
    -------
    int
        0 if the operation was successful, -1 if not supported or failed.
        
    Notes
    -----
    The function automatically detects the operating system and uses the 
    appropriate application. On Windows, it uses the default Excel application.
    On Linux, it uses LibreOffice Calc (soffice).
    """
    ret = 0
    plt = platform.system()
    logger.info(f'Detected {plt} operating system.')
    logger.info(
        f'Closing the file can take up to a minute, wait until the message of successfully edited appears '
        'before pressing any key.'
    )
    
    if plt.lower() == 'windows':
        # os.system(f'start EXCEL.EXE "{wbook_file}"')
        os.system(f'"{wbook_file}"')
    elif plt.lower() == 'linux':
        os.system(f'soffice "{wbook_file}"')
    elif plt.lower() == 'darwin':
        logger.warning('Sorry we do not support MacOS yet.')
    else:
        logger.warning('Unsupported operating system')
        ret = -1
    return ret


def scan_variables(folders):
    """
    Scan Python modules in specified folders to extract self.var variables.
    
    Recursively searches through Python files in the given folders to identify and 
    catalog all self.var variables, tracking their definitions and usage across modules.
    
    Parameters
    ----------
    folders : list of str
        List of folder paths to scan for Python modules.
    
    Returns
    -------
    collections.OrderedDict
        Ordered dictionary mapping variable names to dictionaries containing:
        - Module names as keys with line information as values
        - 'defined' key mapping to the module where the variable is first defined
        Variables are sorted by their defining module.
        
    Notes
    -----
    The function scans all .py files (except List_all_variables.py) in the specified
    folders. For each variable found, it tracks:
    - Which modules define vs. use the variable
    - Line numbers where variables appear
    - The first module that defines each variable
    """

    Dict_AllVariables = {}
    var_def = {}
    for folders_paths in folders:
        path = str(folders_paths)
        python_modules = os.listdir(path)
        for ll in range(len(python_modules)):
            # if the file is a Python module
            if python_modules[ll][-2:] == 'py' and python_modules[ll][-2:] != 'List_all_variables':
                [variable_names_list, associated_line, first_definition] = extract_selfvar(
                    path + '/' + python_modules[ll])
                for ii in range(len(variable_names_list)):  # For each variable
                    if first_definition[ii] == 1:
                        txt_line = 'defined line ' + str(associated_line[ii])
                        var_def[variable_names_list[ii]] = python_modules[ll][:-3]
                    else:
                        txt_line = 'used line ' + str(associated_line[ii])
    
                    name_module_python = python_modules[ll][:-3]  # to go to the line when saving
                    # if the self.var is not already defined
                    if variable_names_list[ii] not in Dict_AllVariables:
                        Dict_AllVariables[variable_names_list[ii]] = {name_module_python: txt_line}
                        if variable_names_list[ii] not in var_def:
                            var_def[variable_names_list[ii]] = name_module_python
    
                    else:  # We add the new module to the list of the variable
                        # if the variable is already defined in this module
                        if name_module_python in Dict_AllVariables[variable_names_list[ii]]:
                            if first_definition[ii] == 1:
                                txt_line = 'updated line ' + str(associated_line[ii])
                            else:
                                txt_line = 'used line ' + str(associated_line[ii])
                            aa = Dict_AllVariables[variable_names_list[ii]][name_module_python]
                            Dict_AllVariables[variable_names_list[ii]][name_module_python] = aa + ', ' + txt_line
                        else:
                            Dict_AllVariables[variable_names_list[ii]][name_module_python] = txt_line
    
    # PB put in where the variable is defined
    
    for ii in Dict_AllVariables:
        Dict_AllVariables[ii]['defined'] = var_def[ii]
    
    # PB sorting by where the variable is defined
    Dict_AllVariables = OrderedDict(sorted(Dict_AllVariables.items(), key=lambda x: getitem(x[1], 'defined')))
    
    return Dict_AllVariables

         

### DEFINING TWO USEFULL FUNCTIONS ###

def extract_selfvar(module_name):
    """
    Extract all 'self.var.varname' variables defined in a Python module.
    
    Parses a Python module file to find all instances of self.var variables,
    tracking their line numbers and whether they represent definitions or usage.
    Handles comment blocks and special cases for the evaporation module.
    
    Parameters
    ----------
    module_name : str
        Full path to the Python module file to analyze.
        
    Returns
    -------
    tuple of (list, list, list)
        A tuple containing:
        - variable_names_list : list of str
            Names of all self.var variables found (e.g., 'self.var.temperature')
        - associated_line : list of int
            Line numbers where each variable appears (1-indexed)
        - first_definition : list of int
            1 if variable is defined/updated on this line, 0 if only used
            
    Notes
    -----
    The function performs several parsing steps:
    - Removes comments and handles multi-line docstrings
    - Special handling for evaporation.py module
    - Identifies variable boundaries using delimiters
    - Distinguishes between variable definitions (line start) and usage
    
    Variable names are extracted by finding 'self.var' patterns and determining
    their boundaries using common Python delimiters like operators, parentheses,
    and method calls.
    """

    # Openning and closing the Python module
    logger.info("----> " + module_name)
    fichier = open(module_name, "r")
    aa = fichier.readlines()
    fichier.close()
    logger.info('Exploring : ' + module_name)
    variable_names_list = []
    associated_line = []  # Line where the module appears
    first_definition = []  # 1 if defined or updated, zero if only used
    test_comment = 0
    for ii in range(len(aa)):  # for each line in the Python code
        bb = aa[ii]
        bb = bb.lstrip()  # Removing space at the beginning

        # test if we are in """ out commented lines:
        # index_out_commented = bb_temp.find('"""')
        # if index_out_commented

        # PB test if the line is a comment line
        bb = bb[:bb.find("#")]

        if bb.find('"""') > -1:
            if test_comment == 1:
                test_comment = 0
            else:
                test_comment = 1
        # sometime start and end """ are in the same line (should not be!): to detect end """ and resume validating
        if bb[3:].find('"""') > -1:
            test_comment = 1

        # This needs to be repaired -- evaporation was not being searched properly
        # With this fix, captures evaporation variables

        if module_name == '../hydrological_modules/evaporation.py':
            test_comment = 0
        # deleted the lines to detect #
        if test_comment == 0:
            indexselfvar = 0
            idselfvar = 0
            while indexselfvar != -1:
                bb_temp = bb[idselfvar:]
                indexselfvar = bb_temp.find("self.var")  # Find all self.var position in the line

                if indexselfvar != -1:  # there is at least one self.var in the line
                    # PB change to +1 and to ww=len(bb_temp) because some variable lost last letter
                    for ww in range(indexselfvar, len(bb_temp) + 1):
                        if (ww == len(bb_temp) or bb_temp[ww] in ':,() =*+-/[]"' or
                                bb_temp[ww:ww + 5] in ['.appe', '.copy', '.ravel'] or bb_temp[ww] in '<>' or
                                bb_temp[ww:ww + 7] == '.astype'):  # Find the end of the var name
                            break
                        # PB changed to ww-indexselfvar because ww is absolut in the line
                        www = ww - indexselfvar
                        if www > 8:
                            if bb_temp[ww] == '.' or bb_temp[ww] == "'":
                                break

                    # Normally, we have defined a 'self.var.xxxxxxx' string
                    if bb_temp[indexselfvar:ww] != 'self.var':
                        # PB tested again because it is not sorted out
                        if bb_temp[indexselfvar:ww] != 'self.var.':
                            variable_names_list.append(bb_temp[indexselfvar:ww])  # Append this variable to the list
                            associated_line.append(ii + 1)  # Append the associated line to the list
                            if indexselfvar == 0:  # if 'self.var' is at the beginning of the line
                                first_definition.append(1)
                            else:
                                first_definition.append(0)
                    idselfvar = idselfvar + indexselfvar + 1

    return variable_names_list, associated_line, first_definition

def extract_localvar(module_name, str_name):
    """
    Extract all local variable names from functions defined in a module.
    
    Uses Python's introspection capabilities to examine function objects and
    extract their local variable names from the code object's co_varnames attribute.
    
    Parameters
    ----------
    module_name : module
        Python module object to analyze.
    str_name : str
        String representation of the module name for building attribute paths.
        
    Returns
    -------
    list of tuple
        List of tuples containing the local variable names for each function
        in the module. Each tuple corresponds to one function's local variables.
        
    Notes
    -----
    This function uses `eval()` to dynamically access function attributes, which
    can be a security concern if used with untrusted input. It examines the
    `__code__.co_varnames` attribute of each function to get variable names.
    
    Only functions that don't start with '__' (dunder methods) are analyzed.
    """

    # Find all functions in the Python module
    sub_fun = [v for v in dir(module_name) if not v.startswith('__')]

    variable_names_list = []
    for ii in range(len(sub_fun)):
        name = str_name + '.' + str(sub_fun[ii]) + '.__code__.co_varnames'
        temp_list = eval(name)
        variable_names_list.append(temp_list)

    return variable_names_list

### EXTRACT LOCAL AND SELF.VAR VARIABLES FROM EACH MODULE


def make_all_variables_df(Dict_AllVariables, df_cur):
    """
    Create a comprehensive DataFrame of all variables by merging discovered and existing data.
    
    Combines newly discovered variables from code analysis with existing variable
    documentation, handling priorities, descriptions, and module information.
    
    Parameters
    ----------
    Dict_AllVariables : dict
        Dictionary mapping variable names to their module usage information.
        Created by scan_variables().
    df_cur : pandas.DataFrame
        Current variable documentation DataFrame with existing descriptions,
        units, and other metadata.
        
    Returns
    -------
    pandas.DataFrame
        Comprehensive DataFrame containing all variables with columns:
        - Variable name, Long name, Unit, Description, Type, Priority
        - First module (where variable is first defined)  
        - Module 1-16 columns showing usage across modules
        New variables get default priority 'low' and empty descriptions.
        
    Notes
    -----
    The function performs several operations:
    - Merges new variables with existing documentation
    - Maintains backward compatibility with legacy Excel files
    - Type can be Array, List, Number or Flag (False/True)
    - Handles missing Priority and Long name columns
    - Sets default values for new variables (priority='low')
    - Preserves existing descriptions and units where available
    
    The resulting DataFrame is structured to match the Excel output format
    with separate columns for each module where variables are used.
    
    """
    df_all_vars = dict(zip(xcols, [[] for i in range(0, len(xcols))]))
    df_cur['Description'] = df_cur['Description'].astype("string")
    df_cur['Type'] = df_cur['Type'].astype("string")
    for k, v in Dict_AllVariables.items():
        if isinstance(v, dict):
            var_name = k.replace('self.var.', '')
            df_all_vars['Variable name'].append(var_name)
            df_all_vars['Unit'].append('')
            df_all_vars['Description'].append(np.NaN)
            df_all_vars['Type'].append(np.NaN)
            df_all_vars['Priority'].append(np.NaN)
            df_all_vars['Long name'].append(np.NaN)
            cnt = 1
            for kv, vv in v.items():
                if kv == 'defined':
                    df_all_vars['First module'].append(vv)
                else:
                    col = f'Module {cnt}'
                    df_all_vars[col].append(f'{kv}: {vv}')
                    cnt += 1
            if cnt < 16:  # fill up Module n columns
                for i in range(cnt, 17):
                    col = f'Module {i}'
                    df_all_vars[col].append('')

    df_all_vars = pd.DataFrame(df_all_vars)

    drop_cols = [c + '_r' for c in xcols[1:]]
    df = df_all_vars.join(df_cur.set_index('Variable name'), rsuffix='_r', on='Variable name')
    df.reset_index(drop=True, inplace=True)

    # use if to maintain retro-compatibility with legacy excel files that do not have priority
    if 'Priority' in list(df_cur.columns):
        df['Priority'] = df['Priority_r']
    else:
        drop_cols.remove('Priority_r')

    if 'Type' in list(df_cur.columns):
        df['Type'] = df['Type_r']
    else:
        drop_cols.remove('Type_r')

    if 'Long name' in list(df_cur.columns):
        df['Long name'] = df['Long name_r']
    else:
        drop_cols.remove('Long name_r') 

    df_new = df[df['Description_r'].isnull()].copy(deep=True)
    df_old = df[~df['Description_r'].isnull()].copy(deep=True)
    df_old['Description'] = df_old['Description_r']
    df_old['Unit'] = df_old['Unit_r']

    df_new_old = pd.concat([df_new, df_old], ignore_index=True)
    # print(df_new_old.columns)

    # print(drop_cols)
    df_new_old.drop(columns=drop_cols, index=1, inplace=True)
    mask = df_new_old['Priority'].isnull()
    df_new_old['Priority'] = np.where(mask, 'low', df_new_old['Priority'])

    mask = df_new_old['Long name'].isnull()
    df_new_old['Long name'] = np.where(mask, '', df_new_old['Long name'])

    mask = df_new_old['Type'].isnull()
    df_new_old['Type'] = np.where(mask, '', df_new_old['Type'])

    return df_new_old

def write_new_excel(wbook_file, df_new_old):
    """
    Write variable data to Excel file and create separate sheets for each priority level.
    
    Creates an Excel workbook with variable documentation, opens it for user editing,
    and then reorganizes the data into separate worksheets based on priority levels.
    
    Parameters
    ----------
    wbook_file : str
        Path to the Excel file to be created/updated.
    df_new_old : pandas.DataFrame
        DataFrame containing all variable information with descriptions,
        units, priorities, and module usage data.
        
    Returns
    -------
    int
        Always returns 0 indicating successful completion.
        
    Notes
    -----
    The function performs several steps:
    1. Writes the complete variable DataFrame to the 'variables' sheet
    2. Opens the file for user editing using the system's default application
    3. After user edits, re-reads the file and creates separate sheets for each priority level
    4. Each priority level (e.g., 'high', 'medium', 'low') gets its own worksheet
    
    The function blocks execution waiting for user input after opening the Excel file,
    allowing manual editing of descriptions, units, and priorities before continuing.
    """
    with pd.ExcelWriter(wbook_file) as writer:
        df_new_old.to_excel(writer, sheet_name='variables', index=False)
        logger.info(f'{wbook_file} file with all variables saved.')

    print('Press enter to open and edit the excel file with the variable. Once edited, save it and close to '
          'continue the variables documentation process')
    wbook_edited = open_workbook(wbook_file=wbook_file)
    if wbook_edited == 0:
        logger.info(f'{wbook_file} successfully edited')
    else:
        logger.warning(f'{wbook_file} was not edited')

    # make one sheet for all priority levels
    df = pd.read_excel(wbook_file, 'variables')
    prio_levels = list(df['Priority'].unique())
    with pd.ExcelWriter(wbook_file) as writer:
        df.to_excel(writer, sheet_name='variables', index=False)

        for l in prio_levels:
            df_l = df[df['Priority'] == l]
            df_l.to_excel(writer, sheet_name=l, index=False)

    return 0

def write_to_metaNetCdf(df_new_old, netxml_file):
    """
    Generate NetCDF metadata XML file from variable documentation DataFrame.
    
    Creates or updates an XML file containing NetCDF metadata for CWATM variables,
    preserving existing standard names while adding new variable definitions.
    
    Parameters
    ----------
    df_new_old : pandas.DataFrame
        Complete variable documentation DataFrame with columns:
        Variable name, Long name, Unit, Description, etc.
    netxml_file : str
        Path to the output XML metadata file to be created/updated.
        
    Returns
    -------
    int
        Always returns 0 indicating successful completion.
        
    Notes
    -----
    The function performs several key operations:
    - Attempts to load existing metadata from the XML file if it exists
    - Preserves existing standard_name attributes for known variables
    - Handles special unit conversions (e.g., '°C' -> 'C' for NetCDF compatibility)
    - Creates CWATM-formatted XML metadata entries for each variable
    - Includes predefined time aggregation metadata in the header
    
    If the existing XML file cannot be parsed, the user is prompted to either
    create a new file ('c') or abort execution ('a').
    
    The output XML follows the CWATM metadata format with entries like:
    <metanetcdf varname="temperature" unit="C" standard_name="air_temperature" 
                long_name="Air temperature" description="..." title="CWATM" author="IIASA WAT" />
    """
    metaNetcdfVar = {}
    user_opt = 'c'
    # load existing info if file exists
    if os.path.isfile(netxml_file):
        # open the metanetcdf file
        try:
            metaparse = minidom.parse(netxml_file)
            meta = metaparse.getElementsByTagName("CWATM")[0]
            for metavar in meta.getElementsByTagName("metanetcdf"):
                d = {}
                for key in list(metavar.attributes.keys()):
                    if key != 'varname':
                        d[key] = metavar.attributes[key].value
                key = metavar.attributes['varname'].value
                metaNetcdfVar[key] = d
        except Exception as e:
            logger.error(f'An error occured while trying to read the {netxml_file} file')
            logger.error(e)
            user_opt = input('Press "c" to create a new empty file or "a" to abort the execution.')
        finally:
            if user_opt == 'a':
                exit()
    with open(netxml_file, 'w') as f:
        f.writelines(netxml_head)

        for _index, row in df_new_old.iterrows():
            var_name = row['Variable name']
            long_name = row['Long name']
            unt = row['Unit']
            if unt == '°C':
                unt = 'C'
            des = row['Description']
            type = str(row['Type'])
            des = des + "[" + type + "]"

            if isinstance(des, pd._libs.missing.NAType):
                des = ''
            standard_name = ''

            if var_name in metaNetcdfVar.keys():
                standar_name = metaNetcdfVar[var_name]['standard_name']

            line = (f'<metanetcdf varname="{var_name}" unit="{unt}"  standard_name="{standard_name}" '
                    f'long_name="{long_name}" description="{des}"  title="CWATM" author="IIASA WAT" />\n')
            f.write(line)

        f.write('</CWATM>')

    return 0


def get_vars_modules_descriptor(df_new_old):
    """
    Extract variable descriptions and units into a lookup dictionary.
    
    Creates a dictionary mapping variable names to their descriptions and units
    for easy lookup during module documentation generation.
    
    Parameters
    ----------
    df_new_old : pandas.DataFrame
        Complete variable documentation DataFrame containing columns:
        'Variable name', 'Description', 'Unit', and other metadata.
        
    Returns
    -------
    dict
        Dictionary with variable names as keys and dictionaries as values.
        Each value dictionary contains:
        - 'Description': Variable description text
        - 'Unit': Variable unit string
        
    Notes
    -----
    This function is used to create a convenient lookup table for variable
    metadata when generating documentation that gets injected into Python
    module docstrings. It extracts only the essential description and unit
    information needed for the module documentation tables.
    """
    # Sort df_new_old by Type (Flag, Number, List, Array) then by Variable name alphabetically
    type_order = {'Flag': 1, 'Number': 2, 'List': 3, 'Array': 4}
    df_sorted = df_new_old.sort_values(by=['Type', 'Variable name'],
                                               key=lambda x: x.map(type_order).fillna(5) if x.name == 'Type' else x)

    new_old = []
    for _index, row in df_sorted.iterrows():
        var_name = row['Variable name']
        uom = row['Unit']
        descr = row['Description']
        type = row['Type']
        new_old.append([var_name,type,descr,uom])


    dict_new_old = {}
    for _index, row in df_sorted.iterrows():
        var_name = row['Variable name']
        uom = row['Unit']
        descr = row['Description']
        type = row['Type']
        dict_new_old[var_name] = {'Type': type,'Description': descr, 'Unit': uom}

    return new_old, dict_new_old

if __name__ == '__main__':
    folders = ['../' + f for f in base_folders]
    # Keys are variable name : then 1rst module and associated line
    Dict_AllVariables = scan_variables(folders)
    kn = len(Dict_AllVariables.keys())
    logger.info(f'Found {kn} variables.')

    # get variables in previous xcel
    wbook_file = input(
        "Enter the excel file name where the variables names and attributes are stored or press any key to "
        "accept the default name (selfvar.xlsx).\nIf the file does not exist, it will be created else "
        "overwritten.\n"
    )

    if len(wbook_file) < 2:
        wbook_file = 'selfvar.xlsx'  # set default if any key pressed
    if 'xlsx' not in wbook_file:
        wbook_file += '.xlsx'

    # create file if not existing
    if not os.path.isfile(wbook_file):
        df = pd.DataFrame(columns=xcols)
        df = pd.DataFrame(df)
        with pd.ExcelWriter(wbook_file) as writer:
            df.to_excel(writer, sheet_name='variables', index=False)
            logger.info(f'{wbook_file} file created.')

    # find all the variables not included in the current documentation
    df_cur = pd.read_excel(wbook_file, sheet_name='variables')
    all_vars = list(Dict_AllVariables.keys())
    all_vars_names = [k.replace('self.var.', '') for k in all_vars]  # all variables names read from the code
    df_all_names = pd.DataFrame({'vars': all_vars, 'var_names': all_vars_names})
    df_new = df_all_names[~df_all_names['var_names'].isin(df_cur['Variable name'])]
    n_nw = len(df_new.index)
    logger.info(f'Found {n_nw} new variables.')

    # print(df_new.head(100))
    # create a dataframe with all variables and save it to excel
    df_new_old = make_all_variables_df(Dict_AllVariables, df_cur)

    # write new excel with al variables and a worksheet for each priority level
    write_new_excel(wbook_file, df_new_old)

    netxml_file = input(
        "Enter the xml file name where the variables NetCDF4 metadata will be stored or press any key to "
        "accept the default name (metaNetcdf.xml).\nIf the file does not exist, it will be created else "
        "overwritten.\n"
    )

    if len(netxml_file) < 2:
        netxml_file = 'metaNetcdf.xml'  # set default if any key pressed
    if 'xml' not in netxml_file:
        netxml_file += '.xml'

    write_to_metaNetCdf(df_new_old, netxml_file)
    
    
    
    print('Modifying CWATM modules to add self.var information')

    for folders_paths in folders:
        path = str(folders_paths)
        python_modules = os.listdir(path)
        for ll in range(len(python_modules)):
            if python_modules[ll][-2:] == 'py':
                filename_to_modify = path + '/' + python_modules[ll]

                # We know the Python file, now we need to find all variables inside
                var_name_in_module = []

                for var_name in Dict_AllVariables:  # FOR EACH SELF.VARIABLE NAME
                    for keys in Dict_AllVariables[var_name]:
                        if python_modules[ll][:-3] == keys:  # we search the module where we want to add information
                            var_name_in_module.append(
                                var_name)  # This list contains all var_name appearing in this module

                if len(var_name_in_module) > 0:

                    ## Now, we modify the Python module to add self.var description
                    logger.info("=== " + filename_to_modify)
                    file = open(filename_to_modify, 'r')
                    lines = file.readlines()
                    file.close()

                    # PB change to make the output table variable
                    lead = " " * 4
                    between = " " * 2
                    col1 = 35
                    col11 = 10
                    col2 = 70
                    col3 = 5
                    # col4 = 30

                    added_description = "\n"+ lead + "**Global variables**\n"

                    added_description = (added_description + lead + '{:{x}.{x}}'.format('=' * col1, x=col1) +
                                         between + '{:{x}.{x}}'.format('=' * col11, x=col11) + between +
                                         between + '{:{x}.{x}}'.format('=' * col2, x=col2) + between +
                                         '{:{x}.{x}}'.format('=' * col3, x=col3) + "\n")
                    added_description = (added_description + lead + '{:{x}.{x}}'.format('Variable [self.var]', x=col1) +
                                         between + '{:{x}.{x}}'.format('Type', x=col11) + between +
                                         between + '{:{x}.{x}}'.format('Description', x=col2) + between +
                                         '{:{x}.{x}}'.format('Unit', x=col3) + "\n")
                    added_description = (added_description + lead + '{:{x}.{x}}'.format('=' * col1, x=col1) +
                                         between + '{:{x}.{x}}'.format('=' * col11, x=col11) + between +
                                         between + '{:{x}.{x}}'.format('=' * col2, x=col2) + between +
                                         '{:{x}.{x}}'.format('=' * col3, x=col3) + "\n")

                    # added_description = added_description + "    ==========================  ========================================================================================  =========  ==============================\n"
                    # added_description = added_description + "    Variable [self.var]         Description                                                                               Unit       Appears in\n"
                    # added_description = added_description + "    ==========================  ========================================================================================  =========  ==============================\n"
                    # put variable names: {description, units} in a dictionary
                    new_old,dict_new_old = get_vars_modules_descriptor(df_new_old)
                    for vv in var_name_in_module:
                        list_modules = ""
                        v_name = vv.replace('self.var.', '')
                        if v_name in dict_new_old.keys():
                            ds_um = dict_new_old[v_name]
                            type = ds_um['Type']
                            descr = ds_um['Description']
                            uim = ds_um['Unit']
                            
                            if isinstance(descr, str) == False:
                                descr = ''
                            if isinstance(uim, str) == False or uim == '-' or len(uim) == 0:
                                uim = '--'

                        added_description = (added_description + lead + '{:{x}.{x}}'.format(vv[9:], x=col1) +
                                             between + '{:{x}.{x}}'.format(type, x=col11) + between +
                                             between + '{:{x}.{x}}'.format(descr, x=col2) + between +
                                             '{:{x}.{x}}'.format(uim, x=col3) + "\n")

                    # added_description = added_description + "    ==========================  ========================================================================================  =========  ==============================\n\n"
                    # added_description = added_description + lead + '{:{x}.{x}}'.format('='*col1,x=col1) + between + '{:{x}.{x}}'.format('='*col2,x=col2) + between +\
                    #                    '{:{x}.{x}}'.format('='*col3,x=col3) + between + '{:{x}.{x}}'.format('='*col4,x=col4) + "\n\n"

                    added_description = (added_description + lead + '{:{x}.{x}}'.format('=' * col1, x=col1) +
                                         between + '{:{x}.{x}}'.format('=' * col11, x=col11) + between +
                                         between + '{:{x}.{x}}'.format('=' * col2, x=col2) + between +
                                         '{:{x}.{x}}'.format('=' * col3, x=col3) + "\n\n")
                    ###added_description = added_description + "    **Functions**\n"

                    # PB delete old list in module (if there is any)
                    linesnew = []
                    add = True
                    for line in lines:
                        if line.find('**Global variables**') > -1:
                            add = False
                        if add:
                            linesnew.append(line)
                        ##if line.find('**Functions**') > -1:
                        ##    add = True
                    lines = linesnew

                    # Find where to add this description
                    class_index = -1
                    begin_outcommented_lines = -1
                    end_outcommented_lines = -1
                    for module_line in range(len(lines)):
                        if lines[module_line].startswith('class'):  # the class is defined here
                            class_index = module_line
                        if lines[module_line].startswith('    """') and class_index != -1:
                            begin_outcommented_lines = module_line
                        if (lines[module_line].startswith('    """') and begin_outcommented_lines != -1 and
                                class_index != -1):
                            end_outcommented_lines = module_line

                    if end_outcommented_lines != -1:  # we found a class and definition just after, so we can add text
                        # lines[end_outcommented_lines-1] = lines[end_outcommented_lines-1] + '\n' + added_description
                        # PB removed the /n in between
                        lines[end_outcommented_lines - 1] = lines[end_outcommented_lines - 1] + added_description

                    # PB to make it unix compatible (windows does not care but linux)
                    file = open(filename_to_modify, mode='w', newline='\n', encoding='utf8')
                    file.writelines(lines)
                    file.close()
                    
    print('Process completed.')
