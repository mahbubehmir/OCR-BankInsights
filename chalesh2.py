import streamlit as st
import cv2
import numpy as np
import pytesseract
import easyocr
from PIL import Image
import re
import pandas as pd
import os
import matplotlib.pyplot as plt

# ----------------- تنظیمات ضروری -----------------
pytesseract.pytesseract.tesseract_cmd = {
    'nt': r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    'posix': '/usr/bin/tesseract',
    'darwin': '/usr/local/bin/tesseract'
}.get(os.name, r'C:\Program Files\Tesseract-OCR\tesseract.exe')

EASYOCR_READER = easyocr.Reader(['fa','en'], gpu=False)

# ----------------- توابع اصلی -----------------
def convert_en_digits(text):
    """تبدیل اعداد انگلیسی به فارسی با حفظ علامت"""
    persian_map = {
        '0':'۰', '1':'۱', '2':'۲', '3':'۳', '4':'۴',
        '5':'۵', '6':'۶', '7':'۷', '8':'۸', '9':'۹',
        '-':'-', ',':',', '.':'.'
    }
    return ''.join(persian_map.get(c, c) for c in str(text))

def enhance_image(image):
    """پردازش تصویر برای بهبود کیفیت OCR"""
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    return Image.fromarray(enhanced)

def ocr_ensemble(image):
    """ترکیب نتایج OCR از دو موتور"""
    text_tesseract = pytesseract.image_to_string(
        image, 
        lang='fas+eng',
        config='--psm 6 --oem 3'
    )
    text_easyocr = '\n'.join(EASYOCR_READER.readtext(np.array(image), detail=0, paragraph=True))
    return f"{text_tesseract}\n{text_easyocr}"

def extract_customer_name(text):
    """استخراج نام مشتری از متن با الگوهای پیشرفته"""
    patterns = [
        r'نام\s*مشتر[ىكی]\s*:\s*(.+)\n',
        r'نام\s*پرداخت\s*کننده\s*:\s*(.+)\n',
        r'customer\s*name\s*:\s*(.+)\n',
        r'نام\s*:\s*(.+)\n'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = convert_en_digits(match.group(1)).strip()
            return name
    return None

def find_customer_records(name, excel_df):
    """جستجوی تمام سوابق مشتری در اکسل"""
    name_columns = ['نام مشتری', 'نام', 'مشتری', 'نام_مشتری', 'ناممشتری']
    target_column = next((col for col in excel_df.columns if col in name_columns), None)
    
    if not target_column:
        raise ValueError("ستون نام مشتری در اکسل یافت نشد")
    
    exact_matches = excel_df[excel_df[target_column].str.contains(name, regex=False, na=False)]
    return exact_matches

# ----------------- رابط کاربری -----------------
st.set_page_config(page_title="جستجوی مشتری", layout="wide")
st.title('🔍 سیستم تحلیل تراکنش های بانکی')

# بخش آپلود فایل‌ها
col1, col2 = st.columns(2)
with col1:
    excel_file = st.file_uploader("فایل اکسل مشتریان (XLSX)", type=['xlsx'])
with col2:
    image_file = st.file_uploader("تصویر فیش بانکی (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if image_file and excel_file:
    try:
        # پردازش اکسل
        excel_df = pd.read_excel(excel_file)
        excel_df = excel_df.astype(str)
        
        # پردازش تصویر
        raw_img = Image.open(image_file)
        processed_img = enhance_image(raw_img)
        
        # استخراج نام از فیش
        with st.spinner('🔍 در حال استخراج نام مشتری...'):
            text = ocr_ensemble(processed_img)
            customer_name = extract_customer_name(text)
            
            if not customer_name:
                raise ValueError("نام مشتری در فیش شناسایی نشد")
                
        # جستجو در اکسل
        with st.spinner('🔎 در حال جستجو در سوابق...'):
            customer_records = find_customer_records(customer_name, excel_df)
        
        # نمایش نتایج مشتری
        st.subheader("نتایج جستجو")
        st.markdown(f"**نام استخراج شده از فیش:** `{customer_name}`")
        
        if not customer_records.empty:
            st.success(f"✅ {len(customer_records)} سوابق مرتبط یافت شد!")
            
            try:
                # تبدیل مبلغ به عددی با حفظ علامت
                customer_records['مبلغ_عددی'] = customer_records['مبلغ'].apply(
                    lambda x: float(re.sub(r'[^-0-9.]', '', x))
                )
                
                # ایجاد ستون جهت تراکنش
                customer_records['جهت تراکنش'] = np.where(
                    customer_records['نوع تراکنش'] == '1', 
                    'واریز', 
                    'برداشت'
                )
                
                # محاسبه آمارهای کلیدی
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                total_deposit = customer_records[customer_records['نوع تراکنش'] == '1']['مبلغ_عددی'].sum()
                total_withdraw = customer_records[customer_records['نوع تراکنش'] == '0']['مبلغ_عددی'].sum()
                net_balance = total_deposit + total_withdraw
                
                with col_stat1:
                    st.metric("مجموع واریزها", f"{total_deposit:,.0f} ریال")
                with col_stat2:
                    st.metric("مجموع برداشت ها", f"{abs(total_withdraw):,.0f} ریال")
                with col_stat3:
                    st.metric("موجودی خالص", f"{net_balance:,.0f} ریال", delta=round(net_balance, 2))
                
                # نمایش خلاصه آماری
                st.subheader("📊 آمارهای تراکنش ها")
                summary = customer_records.groupby('جهت تراکنش').agg(
                    تعداد=('مبلغ', 'count'),
                    میانگین=('مبلغ_عددی', 'mean'),
                    مجموع=('مبلغ_عددی', 'sum')
                ).reset_index()
                
                st.dataframe(summary.style.format({
                    'میانگین': '{:,.0f} ریال',
                    'مجموع': '{:,.0f} ریال'
                }), use_container_width=True)
                
                # تحلیل زمانی
                if 'تاریخ' in customer_records.columns:
                    try:
                        customer_records['تاریخ_تاریخی'] = pd.to_datetime(customer_records['تاریخ'], errors='coerce')
                        
                        # تحلیل ماهانه
                        monthly = customer_records.set_index('تاریخ_تاریخی').groupby(
                            [pd.Grouper(freq='M'), 'جهت تراکنش']
                        )['مبلغ_عددی'].sum().unstack()
                        
                        st.subheader("📈 روند ماهانه تراکنش ها")
                        st.bar_chart(monthly, height=400)
                        
                    except Exception as ex:
                        st.warning("امکان تحلیل زمانی وجود ندارد")
                
                # ----------------- تحلیل‌های اضافی -----------------
                st.subheader("📊 تحلیل‌های اضافی مبالغ تراکنش‌ها")
                
                # تحلیل فراوانی بیشترین مبلغ واریز و برداشت
                deposit_transactions = customer_records[customer_records['نوع تراکنش'] == '1']
                if not deposit_transactions.empty:
                    max_deposit = deposit_transactions['مبلغ_عددی'].max()
                    freq_max_deposit = deposit_transactions[deposit_transactions['مبلغ_عددی'] == max_deposit].shape[0]
                    st.markdown(f"**بیشترین مبلغ واریز:** {max_deposit:,.0f} ریال (تعداد تراکنش: {freq_max_deposit})")
                
                withdraw_transactions = customer_records[customer_records['نوع تراکنش'] == '0']
                if not withdraw_transactions.empty:
                    # در تراکنش‌های برداشت مبلغ به صورت منفی ثبت شده است
                    max_withdraw = withdraw_transactions['مبلغ_عددی'].min()  # کمترین مقدار، بیشترین برداشت محسوب می‌شود
                    freq_max_withdraw = withdraw_transactions[withdraw_transactions['مبلغ_عددی'] == max_withdraw].shape[0]
                    st.markdown(f"**بیشترین مبلغ برداشت:** {abs(max_withdraw):,.0f} ریال (تعداد تراکنش: {freq_max_withdraw})")
                
                # بررسی فراوانی بیشترین تراکنش‌های هزینه و خروجی در یک روز مشخص
                if 'تاریخ_تاریخی' in customer_records.columns:
                    # گروه‌بندی به تفکیک تاریخ (بدون زمان)
                    customer_records['تاریخ_روزانه'] = customer_records['تاریخ_تاریخی'].dt.date
                    daily_summary = customer_records.groupby('تاریخ_روزانه')['مبلغ_عددی'].agg(
                        بیشترین_واریز=lambda x: x[x > 0].max() if any(x > 0) else np.nan,
                        بیشترین_برداشت=lambda x: x[x < 0].min() if any(x < 0) else np.nan
                    ).reset_index()
                
                    if not daily_summary.empty:
                        max_deposit_day = daily_summary.loc[daily_summary['بیشترین_واریز'].idxmax()]
                        max_withdraw_day = daily_summary.loc[daily_summary['بیشترین_برداشت'].idxmin()]
                
                        st.markdown(f"**بیشترین مبلغ واریز در یک روز:** {max_deposit_day['بیشترین_واریز']:,.0f} ریال در تاریخ {max_deposit_day['تاریخ_روزانه']}")
                        st.markdown(f"**بیشترین مبلغ برداشت در یک روز:** {abs(max_withdraw_day['بیشترین_برداشت']):,.0f} ریال در تاریخ {max_withdraw_day['تاریخ_روزانه']}")
                
                    # تحلیل روند ماهانه و هفتگی
                    customer_records['ماه'] = customer_records['تاریخ_تاریخی'].dt.to_period('M').dt.to_timestamp()
                    customer_records['هفته'] = customer_records['تاریخ_تاریخی'].dt.to_period('W').dt.start_time
                
                    monthly_trend = customer_records.groupby('ماه')['مبلغ_عددی'].sum().reset_index()
                    weekly_trend = customer_records.groupby('هفته')['مبلغ_عددی'].sum().reset_index()
                
                    st.subheader("تحلیل روند ماهانه تراکنش‌ها")
                    monthly_chart = monthly_trend.set_index('ماه')
                    st.line_chart(monthly_chart)
                
                    st.subheader("تحلیل روند هفتگی تراکنش‌ها")
                    weekly_chart = weekly_trend.set_index('هفته')
                    st.line_chart(weekly_chart)
                
                    # شناسایی اوج تراکنش‌های مالی به تفکیک ساعت (در صورت وجود اطلاعات زمانی)
                    if customer_records['تاریخ_تاریخی'].dt.hour.nunique() > 1:
                        customer_records['ساعت'] = customer_records['تاریخ_تاریخی'].dt.hour
                        hourly_trend = customer_records.groupby('ساعت')['مبلغ_عددی'].sum()
                        st.subheader("تحلیل تراکنش‌های ساعتی")
                        st.bar_chart(hourly_trend)
                
                    # تحلیل میانگین و انحراف معیار مبالغ تراکنش‌ها
                    mean_amount = customer_records['مبلغ_عددی'].mean()
                    std_amount = customer_records['مبلغ_عددی'].std()
                    st.markdown(f"**میانگین مبالغ تراکنش‌ها:** {mean_amount:,.0f} ریال")
                    st.markdown(f"**انحراف معیار مبالغ تراکنش‌ها:** {std_amount:,.0f} ریال")
                
                # نمایش تمام سوابق
                with st.expander("نمایش تمام سوابق"):
                    st.dataframe(customer_records, use_container_width=True)
                    
                # نمایش متن استخراج شده
                with st.expander("متن کامل استخراج شده از فیش"):
                    st.code(text)
                
            except Exception as e:
                st.error(f'خطا در پردازش داده‌ها: {str(e)}')
        else:
            st.error("❌ هیچ سابقه‌ای برای این مشتری یافت نشد")
    
    except Exception as e:
        st.error(f'⚠️ خطا: {str(e)}')
        
elif image_file and not excel_file:
    st.warning("⚠️ لطفا فایل اکسل مشتریان را نیز آپلود کنید")

st.markdown("---")
st.markdown("### 📖 راهنمای استفاده")
st.markdown("""
1. فایل اکسل حاوی اطلاعات مشتریان را آپلود کنید  
2. تصویر فیش بانکی را آپلود نمایید  
3. سیستم به صورت خودکار:
   - نام مشتری را از فیش استخراج می‌کند
   - سوابق مالی را تحلیل می‌کند
   - آمارهای کلیدی و نمودارهای تعاملی نمایش می‌دهد

**امکانات جدید:**
- نمایش موجودی خالص مشتری
- تفکیک واریز و برداشت
- تحلیل روند ماهانه و هفتگی
- آمارهای توصیفی پیشرفته
- تحلیل فراوانی بیشترین مبالغ و تراکنش‌های هزینه و خروجی در یک روز مشخص
- شناسایی اوج تراکنش‌های مالی (ساعتی)
- محاسبه میانگین و انحراف معیار مبالغ تراکنش‌ها
""")
