import numpy as np
from scipy.interpolate import splprep, splev

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