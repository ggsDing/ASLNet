def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass 
    return False

def read_idtxt(path):
  id_list = []
  #print('start reading')
  f = open(path, 'r')
  curr_str = ''
  while True:
      ch = f.read(1)
      if is_number(ch):
          curr_str+=ch
      else:
          id_list.append(curr_str)
          #print(curr_str)
          curr_str = ''      
      if not ch:
          #print('end reading')
          break
  f.close()
  return id_list