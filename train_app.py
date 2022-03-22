from stream_app import *


def run_params(cmd):
    os.system(cmd)


def training_block():
    st.title("Training ML Predictor")
    type = st.selectbox("Select Model to Train", options=('Valve Position', 'Air Flow'), index=0)
    if type == 'Valve Position':
        option = 'V'
    else:
        option = 'A'
    pred = st.selectbox("Select ML Predictor", options=('lstm', 'gru'), index=0)
    seq = st.number_input("Length of lookback window", min_value=48, max_value=1152, value=48, step=24)
    horizon = st.number_input("Length of prediction horizon", min_value=24, max_value=1152, value=24, step=24)
    params = {
        'seq_length': seq,
        'pred_length': horizon,
        'model': pred,
        'option': option
    }
    base_command = 'python -m prediction.main train'
    for key, value in params.items():
        base_command = base_command+ ' --'+ key + '='+ str(value)
    st.write("Run Command: ")
    st.write(base_command)
    train = st.button("CLICK TO TRAIN")
    if train:
        st.write("Training.....")
        run = run_params(base_command)
        if run:
            st.write( "Successfully trained !")

def main():
    training_block()


if __name__ == "__main__":
    main()
