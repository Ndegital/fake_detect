import streamlit as st
import pandas as pd

#推論用コード
def fake_detect(data):
    worker = []
    company = []
    open = []
    feed_back = []
    for i in range(match_count):
        worker.append(data[5*i+1])
        company.append(data[5*i+2])
        open.append(data[5*i+3])
        feed_back.append(data[5*i+4])
    
    worker = np.array(worker)
    company = np.array(company)
    open = np.array(open)
    feed_back = np.array(feed_back)
    
    # 2. 係数行列・ベクトルの計算
    W = np.zeros((match_count, query_count * 2))
    W[:, 0::2] = (1 + worker) * company
    W[:, 1::2] = (1 - worker) * company
    W /= query_count
    M = np.ones((match_count, query_count * 2+1))
    M[:, 0:query_count * 2] = W
    
    L = np.dot(M.T, M)
    
    # 自滅防止項
    
    add = np.zeros(query_count*2)
    add[0::2] = np.sum((-company/2+0.5)*(worker/2+0.5)*open, axis=0)
    add[1::2] = np.sum((company/2+0.5)*(-worker/2+0.5)*open, axis=0)
    add_1 = np.zeros(query_count*2+1)
    add_1[0:query_count*2] = add
    np.fill_diagonal(L, L.diagonal() + lam2 * add_1)
    
    # 一次の項 c
    ave_company = np.sum(company, axis=1) /  query_count
    vec_target = feedback + ave_company
    c = np.dot(vec_target.T, M) * (-2)
    
    
    add2 = np.zeros(query_count*2+1)
    add2[0::2] = -2*add_1[0::2]
    add2[query_count*2] = 0
    c += lam2*add2
    
    # 3. ニュートン法による最適化
    bias_remove = np.random.rand(query_count * 2 + 1)
    bias_remove[2*query_count] = 0.0
    
    max_newton_iter = 1000
    tol = 1e-6
    
    
    for i in range(max_newton_iter):
        r_params = bias_remove[:-1]
        b_param = bias_remove[-1]
    
        r_clipped = np.clip(r_params, 1e-9, 1.0 - 1e-9)
        x = np.append(r_clipped, b_param)
    
        grad_barrier_r = -1.0/r_clipped + 1.0/(1.0 - r_clipped)
        hess_barrier_r = 1.0/(r_clipped**2) + 1.0/((1.0 - r_clipped)**2)
    
        grad_barrier = np.append(grad_barrier_r, 0)
        hess_barrier = np.append(hess_barrier_r, 0)
    
    
        residuals = np.dot(M, x) - vec_target
        obs_feedback = vec_target - ave_company
    
        mask_lower = (obs_feedback <= -0.99) & (residuals < 0)
        mask_upper = (obs_feedback >= 0.99) & (residuals > 0)
        active_mask = ~(mask_lower | mask_upper)
    
        residuals_masked = residuals * active_mask
    
        grad = 2 * np.dot(M.T, residuals_masked) + lam * grad_barrier
        grad_add = np.zeros(query_count*2)
        grad_add[0::2] = -2 * (1 - r_params[0::2]) * lam2 * np.sum((-company/2+0.5)*(worker/2+0.5)*open, axis=0)
        grad_add[1::2] = 2 * r_params[1::2] * lam2 * np.sum((company/2+0.5)*(-worker/2+0.5)*open, axis=0)
        grad_add_1 = np.zeros(query_count*2+1)
        grad_add_1[0:query_count*2] = grad_add
        grad += grad_add_1
    
        M_masked = M * active_mask[:, np.newaxis]
        L_masked = np.dot(M_masked.T, M)
        np.fill_diagonal(L_masked, L_masked.diagonal() + lam2 * add_1)
    
        H = 2 * L_masked
        H += 0.00001*np.eye(query_count*2+1) #正則化(理論上は正則だが数値エラー防止のため)
        np.fill_diagonal(H, H.diagonal() + lam * hess_barrier)
    
        if np.mean(np.abs(grad)) < tol:
            break
    
        delta = -np.linalg.solve(H, grad)
        step_size = 1.0
        while step_size > 1e-5:
            x_new = x + step_size * delta
            r_new = x_new[:-1]
    
            if np.all((r_new > 0) & (r_new < 1)):
                bias_remove = x_new
                break
            step_size *= 0.5

    bias_remove = bias_remove[0:2*query_count]
    return bias_remove



# タイトル
st.title("虚偽項目推定")

# 1. ファイルアップロード
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### アルゴリズム実行中...")
    result = fake_detect(data)
    df = pd.DataFrame({
    '項目': [i for i in range(1,query_count+1)],
    '虚偽確率': result
    })
    # 3. 出力
    st.write("### 処理結果")
    st.success("計算が完了しました！")
    st.dataframe(result)
