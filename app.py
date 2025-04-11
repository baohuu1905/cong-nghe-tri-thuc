import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from vnstock import *
import plotly.graph_objects as go

from model import PricePredictor
import numpy as np

def display():
    # Get selected values
    symbol = st.session_state.get('selected_company', 'TCB')
    start_date = st.session_state.get('start_date', datetime.today())
    end_date = st.session_state.get('end_date', datetime.today())
    
    # Fetch real Vietnam stock data
    try:
        df = stock_historical_data(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if df is not None and not df.empty:
            # Display OHLC chart
            st.subheader(f"{symbol} Stock Price ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Candlestick(
                x=df['time'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close']
            )])
            fig.update_layout(
                title=f'{symbol} Stock Price',
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show raw data
            st.dataframe(df[['time', 'open', 'high', 'low', 'close', 'volume']])
        else:
            st.warning("Không tìm thấy dữ liệu cho khoảng thời gian này")
    except Exception as e:
        st.error(f"Lỗi khi lấy dữ liệu: {str(e)}")

# Page configuration
st.set_page_config(
    page_title="Streamlit Demo App",
    page_icon="🚀",
    layout="wide"
)


st.title("Thông tin giao dịch chứng khoán")

st.sidebar.title("Nhập dữ liệu chứng khoản")

selected_company = st.sidebar.selectbox(
    "Chọn mã cổ phiếu:",
    ["TCB", "VCB", "ACB", "BID", "CTG", "FPT", "VNM", "VIC"],
    key='selected_company'
)

start_date = st.sidebar.date_input(
    "Chọn ngày bắt đầu:",
    value=datetime.today(),
    key='start_date'
)

st.sidebar.write("Ngày bắt đầu: ", start_date)

end_date = st.sidebar.date_input(
    "Chọn ngày kết thúc:",
    value=datetime.today(),
    key='end_date'
)

st.sidebar.write("Ngày kết thúc: ", end_date)



# Display stock data by default
if st.sidebar.button("Xem thông tin dữ liệu"):
    display()
    # st.experimental_rerun()




predictor = PricePredictor()

st.sidebar.title("Nhập thông tin giao dịch")

# Training section
if st.sidebar.button("Huấn luyện mô hình dự đoán"):
    with st.spinner("Đang huấn luyện mô hình..."):
        predictor.train_model(symbol=st.session_state.get('selected_company', 'TCB'))
    st.sidebar.success("Mô hình đã được huấn luyện!")
    st.sidebar.success("Mã công ty: " + selected_company + "\n\nMô hình dự đoán: LinerRegression()" + "\n\nMô hình chuẩn hóa: MinMaxScaler()")

    
# Prediction inputs
st.sidebar.subheader("Dự đoán giá cổ phiếu")
open_val = st.sidebar.number_input("Open", value=0.0)
high_val = st.sidebar.number_input("High", value=0.0)
low_val = st.sidebar.number_input("Low", value=0.0)
volume_val = st.sidebar.number_input("Volume", value=0.0)

# Initialize prediction history in session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = pd.DataFrame(columns=['time', 'symbol', 'predicted_price'])

# Create two columns for buttons
col1, col2 = st.sidebar.columns(2)

with col1:
    predict_btn = st.button(
        "**Dự đoán Giá**",
        help="Nhấn để dự đoán giá cổ phiếu",
        type="primary",
        use_container_width=True
    )

with col2:
    reset_btn = st.button(
        "Xóa lịch sử",
        help="Xóa toàn bộ lịch sử dự đoán",
        type="secondary",
        use_container_width=True
    )

if predict_btn:
    with st.spinner('Đang thực hiện dự đoán...'):
        try:
            predicted_price = predictor.predict_price(open_val, high_val, low_val, volume_val)
            st.success(f"💰 Giá dự đoán cho {selected_company}: {predicted_price:,.2f} VND")
            
            # Save to history
            new_pred = pd.DataFrame({
                'time': [datetime.now()],
                'symbol': [selected_company],
                'predicted_price': [predicted_price]
            })
            st.session_state.prediction_history = pd.concat(
                [st.session_state.prediction_history, new_pred],
                ignore_index=True
            )
            
            # Show history chart and table
            if not st.session_state.prediction_history.empty:
                st.subheader("📊 Lịch sử dự đoán")
                
                # Display all predictions in a table
                st.dataframe(
                    st.session_state.prediction_history.sort_values('time', ascending=False),
                    column_config={
                        "time": "Thời gian",
                        "symbol": "Mã CP", 
                        "predicted_price": st.column_config.NumberColumn(
                            "Giá dự đoán (VND)",
                            format="%,.2f"
                        )
                    },
                    use_container_width=True
                )
                
                # Show box plot visualization
                fig = px.box(
                    st.session_state.prediction_history,
                    x='symbol',
                    y='predicted_price',
                    color='symbol',
                    labels={'predicted_price': 'Giá (VND)', 'symbol': 'Mã CP'},
                    title=f'Phân bố giá dự đoán cho {selected_company}'
                )
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Lỗi: {str(e)}")

if reset_btn:
    st.session_state.prediction_history = pd.DataFrame(columns=['time', 'symbol', 'predicted_price'])
    st.sidebar.success("Đã xóa lịch sử dự đoán!")
    st.experimental_rerun()


