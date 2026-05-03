import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation

def compute_kappa(pos, s=0.1):
    print("COMPUTE KAPPA..")
    eps = 1e-8

    jitter = np.random.normal(0, 1e-9, pos.shape)
    pos_jittered = pos + jitter

    tck, u = splprep(pos_jittered.T, s=s, k=3)
    
    # 4. גזירה אנליטית
    v_list = splev(u, tck, der=1) 
    a_list = splev(u, tck, der=2) 
    
    v = np.array(v_list).T
    a = np.array(a_list).T
    
    # 5. חישוב Kappa
    cross_product = np.cross(v, a)
    cp_norm = np.linalg.norm(cross_product, axis=1)
    v_norm = np.linalg.norm(v, axis=1)
    
    kappa = cp_norm / (v_norm**3 + eps)

    kappa = kappa[:,np.newaxis]
    
    return kappa


def compute_tau(pos, s=0.1):
    print("COMPUTE TAU..")

    eps_stability = 1e-8
    
    jitter = np.random.normal(0, 1e-9, pos.shape)
    pos_jittered = pos + jitter


    tck, u = splprep(pos_jittered.T, s=s, k=4)

    v_list = splev(u, tck, der=1) 
    a_list = splev(u, tck, der=2) 
    j_list = splev(u, tck, der=3) 
    
    v = np.array(v_list).T
    a = np.array(a_list).T
    j = np.array(j_list).T
    
    # 4. חישוב הטורסיה לפי הנוסחה: [ (v x a) . j ] / |v x a|^2
    
    cross_va = np.cross(v, a)
    
    numerator = np.einsum('ij,ij->i', cross_va, j)
    
    denominator = np.linalg.norm(cross_va, axis=1)**2

    tau = numerator / (denominator + eps_stability)

    return tau[:, np.newaxis]
          
def compute_lambda(R):
    
    print("COMPUTE LAMBDA..")
    eps = 10**(-5)

    R_3_3_vec = R.reshape(-1, 3, 3)
    r_vec = Rotation.from_dcm(R_3_3_vec)
    quat = r_vec.as_quat()

    # 1. compute q_2tag

    q_cur_vec = quat[1:-1,:].copy()
    q_prev_vec = quat[0:-2,:].copy()
    q_next_vec = quat[2:,:].copy()

    # 1.2 compute l 
    # 1.2.1 compute quaternion dot product
    q_dot_prev_cur = np.einsum('ij,ij->i', q_prev_vec, q_cur_vec).reshape(-1,1)
    q_dot_cur_next = np.einsum('ij,ij->i', q_cur_vec, q_next_vec).reshape(-1,1)
    
    temp1 = np.arccos(np.clip(q_dot_prev_cur,-1,1))
    temp2 = np.arccos(np.clip(q_dot_cur_next,-1,1))
    l = temp1/2 + temp2/2

    denominator = l**2
    numerator = q_prev_vec-2*q_cur_vec+q_next_vec

    q_2tag = numerator/(denominator+eps)

    q_2tag_dot_q = np.einsum('ij,ij->i', q_2tag, q_cur_vec).reshape(-1,1)

    q_2tag_dot_q__scalarMulti_q = q_cur_vec*q_2tag_dot_q

    temp = q_2tag_dot_q - q_2tag_dot_q__scalarMulti_q

    _lambda = np.sqrt(np.einsum('ij,ij->i', temp, temp).reshape(-1,1))

    return _lambda



    


if __name__ == '__main__':

    R = 10
    theta = np.linspace(0, 2*np.pi, 100) 
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    z = np.zeros_like(theta) 

    pos_test = np.column_stack((x, y, z)) 

    # 2. הרצת הפונקציה
    try:
        kappa_results = compute_kappa(pos_test, s=0) # s=0 כי המעגל מושלם
        
        # 3. בדיקת תוצאות
        expected_kappa = 1/R # אמור להיות 0.1
        avg_kappa = np.mean(kappa_results)
        
        print("--- Test Results ---")
        print("Expected Kappa: {:.4f}".format(expected_kappa))
        print("Average Calculated Kappa: {:.4f}".format(avg_kappa))
        print("Max Deviation: {:.6f}".format(np.max(np.abs(kappa_results - expected_kappa))))
        
    except Exception as e:
        print("Test Failed with error:", e)