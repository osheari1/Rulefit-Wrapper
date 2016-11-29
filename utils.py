import logging
import os
import re
import numpy as np
from pprint import pprint


def get_logger(log_path):# {{{
  logger = logging.getLogger(__name__)
  
  if os.path.exists(os.path.join(log_path)):
    os.remove(os.path.join(log_path))
  file_hdler = logging.FileHandler(log_path)
  formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
  file_hdler.setFormatter(formatter)
  logger.addHandler(file_hdler)
  return logger# }}}

def parse_rules(path):# {{{
  """ Function that parses the output given by the R rulefit rules function and
      returns the rules in a dictionary format.
  Returns:
    rules - A dictionary containing ranked rules. An example of the two types
            of outputs are shown below.
            {
              'Rule 72': {'info': {'coeff': 0.002348, 'importance': 0.07413, 'support': 0.2806},
                          'type': 'rule',
                          'vars': {'excl': {'lstat_cat': [3]},
                                   'incl': {'lstat_cat': (1, 2, 3)},
                                   'range': {'rm': (6.062, inf)}}}

              'Rule 4': {'info': {'coeff': 0.1041, 'impotance': 71.33, 'std': 9.747},
                         'type': 'lin',
                         'vars': 'rad'}
            }
  """
  with open(path) as f:
    rules_in = f.readlines()
   
  # First three lines are useless
  rules_in = rules_in[3:]
  rules_l = []
  rules = {} 
  current_rule = ''
  info_done = False
  range_done = False 
  prev_line = None
  for i, line in enumerate(rules_in):
    line = re.split('\t|\n|\r| ',line.replace(':', ' ').replace('=', ' '))
    line = [token for token in line if token != '']

    # Remove empty lines
    if len(line) == 0:  
      continue

    # New rule
    elif line[0] == 'Rule':  
      current_rule = int(line[1])
      rules[current_rule] = {'info': {}}
      if line[-1] != 'variables':
        rules[current_rule]['type'] = 'lin'
        rules[current_rule]['vars'] = line[-1]
      else:
        rules[current_rule]['type'] = 'rule'
        # rules[current_rule]['vars'] = {'range': {}, 'incl': {}, 'excl': {}}     
        rules[current_rule]['vars'] = {}
      info_done = False
      range_done = False

    # Describes rule statistics 
    elif line[0] == 'support' or line[0] == 'std':  
      rules[current_rule]['info'] = {line[0]: float(line[1]),
                                     line[2]: float(line[3]),
                                     line[4]: float(line[5])}
      info_done = True

    # Get categorical variables
    elif info_done and prev_line and prev_line[-1] == 'in' \
                                  and prev_line[-2] != 'not':
      if not 'incl' in rules[current_rule]['vars'].keys():
        rules[current_rule]['vars'].update({'incl': {}})

      rules[current_rule]['vars']['incl'].update(
          {prev_line[0]: [int(float(l)) for l in line]}
          )
      range_done = True

    elif info_done and prev_line and prev_line[-1] == 'in'  \
                                  and prev_line[-2] == 'not':
      if not 'excl' in rules[current_rule]['vars'].keys():
        rules[current_rule]['vars'].update({'excl': {}})

      rules[current_rule]['vars']['excl'].update(
          {prev_line[0]: [int(float(l)) for l in line]}
          )
      range_done = True

    # Get numerical variables
    elif info_done and not range_done and line[-1] != 'in':  
      if not 'excl' in rules[current_rule]['vars'].keys():
        rules[current_rule]['vars'].update({'range': {}})
      rules[current_rule]['vars']['range'] \
          .update({line[0]: (-np.inf if line[-2] == '-0.9900E+36'
                                     else float(line[-2]),
                              np.inf if line[-1] == '0.9900E+36'
                                     else float(line[-1]))})
    
    # Add tracker for previous line
    prev_line = line  
  return rules # }}}


# if __name__ == '__main__':
  # rules_path = \
  # '/home/riley/R/x86_64-pc-linux-gnu-library/3.3/rulefit/rulesout.hlp'
  # pprint(parse_rules(rules_path))
   
