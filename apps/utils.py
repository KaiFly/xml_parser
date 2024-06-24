data_config = {
    "1": {"Thẻ": "HDon/DLHDon/TTChung", "Chỉ tiêu": "KHMSHDon", "Mô tả": "Ký hiệu mẫu số hóa đơn"},
    "2": {"Thẻ": "HDon/DLHDon/TTChung", "Chỉ tiêu": "KHHDon", "Mô tả": "Ký hiệu hóa đơn"},
    "3": {"Thẻ": "HDon/DLHDon/TTChung", "Chỉ tiêu": "SHDon", "Mô tả": "Số hóa đơn"},
    "4": {"Thẻ": "HDon/DLHDon/TTChung", "Chỉ tiêu": "NLap", "Mô tả": "Ngày lập"},
    "5": {"Thẻ": "HDon/DLHDon/TTChung/TTHDLQuan", "Chỉ tiêu": "TCHDon", "Mô tả": "Tính chất hóa đơn"},
    "6": {"Thẻ": "HDon/DLHDon/NDHDon/NBan", "Chỉ tiêu": "Ten", "Mô tả": "Tên người bán"},
    "7": {"Thẻ": "HDon/DLHDon/NDHDon/NBan", "Chỉ tiêu": "MST", "Mô tả": "Mã số thuế người bán"},
    "8": {"Thẻ": "HDon/DLHDon/NDHDon/TToan", "Chỉ tiêu": "TgTCThue", "Mô tả": "Tổng tiền (chưa có thuế GTGT)"},
    "9": {"Thẻ": "HDon/DLHDon/NDHDon/TToan", "Chỉ tiêu": "TgTThue", "Mô tả": "Tổng tiền thuế GTGT"},
    "10": {"Thẻ": "HDon/DLHDon/NDHDon/TToan", "Chỉ tiêu": "TgTTTBSo", "Mô tả": "Tổng tiền thanh toán bằng số"}
}

data_tron_thue_xls_path = "data_input/DN_TRON_THUE.xlsx"
data_tron_thue_csv_path = "data_input/DN_TRON_THUE.csv"


def get_nested(data, args):
    """
    get data from dictionary based on tree structured args
    """
    if args and data:
        element  = args[0]
        if element:
            value = data.get(element)
            return value if len(args) == 1 else get_nested(value, args[1:])

def to_excel(df):
    """
    Convert dataframe to excel type ready for streamlit download
    """
    from io import BytesIO
    in_memory_fp = BytesIO()
    df.to_excel(in_memory_fp)
    in_memory_fp.seek(0, 0)
    return in_memory_fp.read()


def get_leaves_and_parse(dct):
    """    
    Extracts all leaves from a nested dictionary and returns a list of leaves and a parsed dictionary.
    """
    leaves = []
    dct_parse = {}

    def recursive_extract(d, path=''):
        if isinstance(d, dict):
            for key, value in d.items():
                new_path = f"{path}.{key}" if path else key
                recursive_extract(value, new_path)
        else:
            leaves.append(d)
            dct_parse[path] = d

    recursive_extract(dct)
    return leaves, dct_parse


## Reformat currency data
def currency_format(x):
    return "{:,.0f}".format(x)


def currency_format2(x):
    return "{:,.1f}".format(x)


def round_money(money, round_number):
    if np.isnan(money):
        return money
    base = 10**round_number
    return base * round(money / base)


def normalize_currency(money, round_number=6, is_decimal=False):
    # round_number = -1, auto select
    if round_number == -1:
        if math.log10(money) >= 9:
            round_number = 9
        elif math.log10(money) >= 6:
            round_number = 6
        else:
            round_number = 3
    base = 10**round_number
    # rounded = base * round(money/base)
    dict_prefix = {3: "K", 6: "M", 9: "B"}
    if is_decimal == False:
        res = f"{currency_format(round(money/base))}{dict_prefix[round_number]}"
    else:
        res = f"{currency_format2(round(money/base, 1))}{dict_prefix[round_number]}"
    return res


## Reformat number data
def round_number(number, decimal=0):
    if np.isnan(number):
        return number
    else:
        return round(number, decimal)


def normalize_number(number):
    return "{:,.0f}".format(number)

