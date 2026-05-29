def rare_chars_european(text, lang):
  european_langs = ['en', 'es', 'de', 'fr']
  if lang not in european_langs:
    return False

  basic = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
  punctuation = ".,;:!?()[]{}-–'\"/\\"

  valid_chars = set(basic + punctuation + " \t\n")

  if lang == 'es':
    valid_chars.update("áéíóúüñÁÉÍÓÚÜÑ")
  elif lang == 'de':
    valid_chars.update("äöüßÄÖÜẞ")
  elif lang == 'fr':
    valid_chars.update("àâçéèêëîïôùûüÿÀÂÇÉÈÊËÎÏÔÙÛÜŸ")
  elif lang == 'en': # inglés, no añade nada extra
    pass

  for c in text:
    if c not in valid_chars:
      return True
    return False