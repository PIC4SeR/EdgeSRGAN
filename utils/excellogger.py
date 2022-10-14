from googleapiclient.discovery import build
from google.oauth2 import service_account
import numpy as np
    
class Logger():
    def __init__(self, spredsheet, key='excel_key.json'):
        scopes = ['https://www.googleapis.com/auth/spreadsheets']
        credentials = service_account.Credentials.from_service_account_file(key, scopes=scopes)
        service = build('sheets', 'v4', credentials=credentials)
        self.service = service.spreadsheets()
        self.spredsheet = spredsheet
    
    
    def write(self, data, sheet, row, column, row_end=None, column_end=None):
        column = Column(column)
        if row_end is None:
            row_end = row
        if column_end is None:
            column_end = column
        shape = self._get_shape(row, column, row_end, column_end)
        data = np.array(data).reshape(*shape).tolist()
        request = self.service.values().update(spreadsheetId=self.spredsheet,
                                               range=f"{sheet}!{column}{row}:{column_end}{row_end}",
                                               valueInputOption="USER_ENTERED", body={"values":data}).execute()
        
    def read(self, sheet, row, column, row_end=None, column_end=None):
        column = Column(column)
        if row_end is None:
            row_end = row
        if column_end is None:
            column_end = column
        result = self.service.values().get(spreadsheetId=self.spredsheet,
                                           range=f"{sheet}!{column}{row}:{column_end}{row_end}").execute()
        values = result.get('values', [])
        return values
    
            
    def _get_shape(self, row, column, row_end=None, column_end=None):
        if not isinstance(column, Column):
            column = Column(column)
        if not isinstance(column_end, Column):
            column_end = Column(column_end)
        assert row_end>=row, column_end>=column
        rows = row_end - row + 1
        columns = int(column_end + 1 - column)
        return (rows, columns)  
            
            
            
class Column():
    def __init__(self, value):
        self.value = self._check_valid(value)
            
    def _check_valid(self, value):
        if not isinstance(value, (str, int, self.__class__)):
            raise ValueError(f"Column type should be int or string, got {type(value)}.")
        if isinstance(value, int):
            value = self._convert(value)
        if isinstance(value, self.__class__):
            value = value.value
        return value
        
    def _convert(self, int_value):
        if int_value<1:
            raise ValueError(f"Column number should be positive, got {int_value}.")
        result = ''
        while int_value > 0:
            index = (int_value - 1) % 26
            result += chr(index + 65)
            int_value = (int_value - 1) // 26
        return result[::-1]
    
    def _reverse_convert(self, string_value):
        string_value = string_value.upper()
        result = 0
        i = 0 
        while len(string_value) > 0:
            v = ord(string_value[-1]) - ord('A') + 1
            result += v*(26**i)
            i+=1
            string_value = string_value[:-1]
        return result
    
    def __str__(self):
        return self.value.upper()
    
    def __int__(self):
        return self._reverse_convert(self.value)
    
    def __repr__(self):
        return str(self)
    
    def __add__(self, other):
        if other == 0:
            return self
        other = self._check_valid(other)
        return  Column(self._reverse_convert(self.value) +  self._reverse_convert(other))
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        if other == 0:
            return self
        other = self._check_valid(other)
        return Column(self._reverse_convert(self.value) -  self._reverse_convert(other))
    
    def __rsub__(self, other):
        return self - other
    
    def __gt__(self, other):
        return int(self) > int(other)

    def __ge__(self, other):
        return int(self) >= int(other)

    def __lt__(self, other):
        return int(self) < int(other)

    def __le__(self, other):
        return int(self) <= int(other)

    def __eq__(self, other):
        try:
            a = int(self)
            b = int(other)
        except:
            return False
        return a == b

    def __ne__(self, other):
        return int(self) != int(other)
 
    
