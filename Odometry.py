from glob import glob
import cv2, skimage, os
import numpy as np

class OdometryClass:
    def __init__(self, frame_path):
        self.frame_path = frame_path
        self.frames = sorted(glob(os.path.join(self.frame_path, 'images', '*')))
        with open(os.path.join(frame_path, 'calib.txt'), 'r') as f:
            lines = f.readlines()
            self.focal_length = float(lines[0].strip().split()[-1])
            lines[1] = lines[1].strip().split()
            self.pp = (float(lines[1][1]), float(lines[1][2]))
            
        with open(os.path.join(self.frame_path, 'gt_sequence.txt'), 'r') as f:
            self.pose = [line.strip().split() for line in f.readlines()]


        # camera matrix
        self.K = np.array([
                        [self.focal_length,0,self.pp[0]],
                        [0, self.focal_length, self.pp[1]],
                        [0,0,1]
                ])

        # initialize feature detector
        self.detector = cv2.ORB_create(3000)

        # initialize feature matcher 
        # adapted from open cv documentation https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=100)
        self.matcher = cv2.FlannBasedMatcher(index_params,search_params)
        
    def imread(self, fname):
        """
        read image into np array from file
        """
        return cv2.imread(fname, 0)

    def imshow(self, img):
        """
        show image
        """
        skimage.io.imshow(img)

    def get_gt(self, frame_id):
        pose = self.pose[frame_id]
        x, y, z = float(pose[3]), float(pose[7]), float(pose[11])
        return np.array([[x], [y], [z]])

    def get_rt_0(self):
        pose = self.pose[0]
        x, y, z = float(pose[3]), float(pose[7]), float(pose[11])
        t = np.array([[x], [y], [z]])
        rl = pose[:3] + pose[4:7] + pose[8:11]
        R = [float(x) for x in rl]
        R = np.array(R).reshape(3,3)

        return R,t

    def get_scale(self, frame_id):
        '''Provides scale estimation for mutliplying
        translation vectors
        
        Returns:
        float -- Scalar value allowing for scale estimation
        '''
        prev_coords = self.get_gt(frame_id - 1)
        curr_coords = self.get_gt(frame_id)
        return np.linalg.norm(curr_coords - prev_coords)

    def get_kps_data(self): 
        """
        ---------- ADAPTED FROM PS3 --------------
        detect the keypoints and compute their descriptors with opencv library
        """

        kps = []
        des = []
        for filename in self.frames:
            img = self.imread(filename)
            kp, descriptors = self.detector.detectAndCompute(img, None) 
            kps.append(kp)
            des.append(descriptors)

        return kps, des

    def get_matches(self,kp1,des1,kp2,des2,num_matches=50):

        """

        Function to get the best matches from the two frames
        
        Returns:
        l_pts_2d: 2xn - points in the left image
        r_pts_2d: 2xn - points in the right image
        """
        
        # match
        try:
            matches = self.matcher.knnMatch(des1,des2,k=2)
            # Lowe's ratio test
            good_matches = []
            ratio = 0.75
            for tpl in matches:
                try:
                    m,n = tpl
                    if m.distance < ratio*n.distance:
                        good_matches.append(m)
                except:
                    continue
        except:
            good_matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des1,des2)
            

        if len(good_matches) > num_matches:
            # Sort the matches and only take the top num_matches ones
            good_matches = sorted(good_matches, key=lambda x: x.distance)
            good_matches = good_matches[:num_matches]

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        return pts1.T, pts2.T 

    def compute_inliers(self,l_pts_2d,r_pts_2d,F,thresh):
        """
        F: 3x3
        l_pts_2d: 2xn
        r_pts_2d: 2xn
        """


        #  convert to homogenous coords
        n = l_pts_2d.shape[1]
        ones = np.ones((1,n))
        l_pts_h = np.vstack([l_pts_2d,ones]) # 3xn
        r_pts_h = np.vstack([r_pts_2d,ones]) # 3xn


        errs = np.zeros((n,))
        for i in range(n):
            err = (r_pts_h[:,i:i+1].T @ F) @ l_pts_h[:,i:i+1] # 1x1
            errs[i] = abs(err[0])


        inlier_indices = np.where(errs <= thresh)[0]
        num_inliers = len(inlier_indices)

        inlier_set_l = None 
        inlier_set_r = None
        if num_inliers > 0:
            inlier_set_l = l_pts_2d[:,inlier_indices]
            inlier_set_r = r_pts_2d[:,inlier_indices]

        return num_inliers,inlier_set_l,inlier_set_r



    def get_fundamental_matrix_ransac(self,l_pts_2d,r_pts_2d,K,s=8,max_iters=1000,min_inliers=8,error_thresh=0.01):
        """
        l_pts_2d: 2 x n
        r_pts_2d: 2 x n
        """

        final_F = None
        max_inliers = 0
        best_inlier_set_l = None
        best_inlier_set_r = None


        for i in range(max_iters):
            # choose 8 points
            choices = np.random.choice(l_pts_2d.shape[1],size=s,replace=True)
            pl = l_pts_2d[:,choices]
            pr = r_pts_2d[:,choices]
            # compute estimate F
            F = self.compute_fundamental_matrix_normalized(pl,pr)
            # find inliers
            num_inliers,inlier_set_l, inlier_set_r = self.compute_inliers(l_pts_2d,r_pts_2d,F,error_thresh)
            # if inliers sufficient -> recompute 
            if num_inliers >= min_inliers and num_inliers >= max_inliers:
                best_inlier_set_l = inlier_set_l
                best_inlier_set_r = inlier_set_r
                max_inliers = num_inliers

        if best_inlier_set_l is not None:
            final_F = self.compute_fundamental_matrix_normalized(best_inlier_set_l,best_inlier_set_r)
            return final_F,best_inlier_set_l,best_inlier_set_r
        else:
            final_F = self.compute_fundamental_matrix_normalized(l_pts_2d,r_pts_2d)
            return  final_F,l_pts_2d,r_pts_2d

    def compute_fundamental_matrix(self,l_pts_h,r_pts_h):

        A = []

        for i in range(l_pts_h.shape[1]):

            xl,yl,zl = l_pts_h[0,i],l_pts_h[1,i],l_pts_h[2,i]
            xr,yr,zr = r_pts_h[0,i],r_pts_h[1,i],r_pts_h[2,i]

            A.append( [
               xr * xl, xr * yl, xr*zl, yr * xl, yr * yl, yr*zl, zr* xl, zr*yl, zr*zl
            ])

        A = np.array(A)
        U,S,V = np.linalg.svd(A.T @ A)
        F = V[-1,:]

        F = F.reshape(3,3)

        # enforce rank 2 
        U,S,V = np.linalg.svd(F)
        S[2] = 0
        F = U @ np.diag(S) @ V

        return F

    def convert_to_h(self,pts):

        n = pts.shape[1]
        ones = np.ones((1,n))
        pts_h = np.vstack([pts,ones])

        return pts_h

    def normalize_points(self,pts_2d):

        # convert to homogeneous
        n = pts_2d.shape[1]
        ones = np.ones((1,n))
        pts_h = np.vstack([pts_2d,ones])

        # normalize points
        scale = np.sqrt(2) / np.std(pts_2d)
        means = np.mean(pts_2d,axis=1)
        T = np.array([
            [scale, 0 , -scale * means[0]],
            [0, scale , -scale * means[1]],
            [0,      0,               1]
        ])

        pts_h_norm = T @ pts_h

        return pts_h_norm, T


    def compute_fundamental_matrix_normalized(self,l_pts_2d,r_pts_2d):
        
        # normalize points

        l_pts_h_norm , transformation_l = self.normalize_points(l_pts_2d)
        r_pts_h_norm , transformation_r  = self.normalize_points(r_pts_2d)

        # compue F
        F = self.compute_fundamental_matrix(l_pts_h_norm,r_pts_h_norm)

        # denormalize F
        F = transformation_r.T @ F @ transformation_l

        try:
            F = F / F[2,2]
        except:
            return F 

        return F
    

    def get_projection_matrix(self,R,t):

        Rt = np.concatenate([R,t],axis=-1)

        return self.K @ Rt

    def traingulate(self,l_pts_2d,proj_l, r_pts_2d, proj_r):
        
        """
        ----------------- ADAPTED FROM PS4 ---------------
        """
        l_pts_2d = l_pts_2d.T
        r_pts_2d = r_pts_2d.T

        pts = []
        for i in range(l_pts_2d.shape[0]): 
            A = []
            row_1 = [
                (l_pts_2d[i,1] * proj_l[2,:]) - proj_l[1,:]
            ]
            row_2 = [
                proj_l[0,:] - (l_pts_2d[i,0]*proj_l[2,:])
            ]

            A.append(row_1)
            A.append(row_2)

            row_1 = [
                (r_pts_2d[i,1] * proj_r[2,:]) - proj_r[1,:]
            ]
            row_2 = [
                proj_r[0,:] - (r_pts_2d[i,0]*proj_r[2,:])
            ]

            A.append(row_1)
            A.append(row_2)

            A = np.array(A)
            A = A.reshape(A.shape[0],A.shape[2])

            U,S,V = np.linalg.svd(A.T @ A) 
            X = V[-1,:]
            if X[-1] != 0:
                X = X * (1 / X[-1] )

            pts.append(X[:-1])

        pts = np.array(pts)
        return pts.T
        
    def count_points_in_front(self,pts_3d_l,R,t):
        """
        pts_3d: 3xn
        R: 3x3
        t: 3x1
        """

        T = self.form_transformation_matrix(R,t)
        pts_3d_l_h = self.convert_to_h(pts_3d_l)
        pts_3d_r_h = T @ pts_3d_l_h
        pts_3d_r = pts_3d_r_h[:3,:] / pts_3d_r_h[3,:]

        return sum(pts_3d_l[2,:] > 0) + sum(pts_3d_r[2,:] > 0)

    
    def form_transformation_matrix(self,R,t):

        T = np.concatenate([R,t],axis = -1) # 3x4
        T = np.concatenate([T,np.array([[0,0,0,1]])],axis=0) # 4x4

        return T

    def get_best_R_t(self,poses,l_pts_2d,r_pts_2d):
        """
        poses: list of R,t tuples  [(R,t),(R,t),....]
        """

        # P1 = K[I|0]
        P1 = self.get_projection_matrix(np.identity(3),np.zeros((3,1)))

        # perform triangulation using all 4 projection matrices
        counts = []
        for R,t in poses:
            # P2 = K[R|t]
            P2 = self.get_projection_matrix(R,t)
            pts_3d = self.traingulate(l_pts_2d,P1,r_pts_2d,P2)

            # count points in front of cameras for this configuration
            points_in_front = self.count_points_in_front(pts_3d,R,t)
            counts.append(points_in_front)

        # get the configuration with most points in front of the camera
        ## Maybe this is the improvement to simple visual odometry where
        ## improving the estimate of R and t by taking many reconstructed points
        return poses[np.argmax(counts)]


    def decompose_E(self,E,l_pts_2d,r_pts_2d):
        
        # Implementing the method described in Hartley & Zisserman

        U,S,Vt = np.linalg.svd(E)

        W = np.array([
            [0,-1,0],
            [1,0,0],
            [0,0,1]
        ])


        t1 = U[:,2][:,np.newaxis]
        t2 = -t1

        R1 = U @ W @ Vt
        R2 = U @ W.T @ Vt


        poses = [(R1,t1),(R2,t1),(R1,t2),(R2,t2)]

        # check determinants
        for i,pose in enumerate(poses):
            R,t = pose
            if np.linalg.det(R) < 0:
                poses[i] = (-R,-t)


        R,t = self.get_best_R_t(poses,l_pts_2d,r_pts_2d)
        return R,t


    def run(self):
        """
        Uses the video frame to predict the path taken by the camera
        
        The reurned path should be a numpy array of shape nx3
        n is the number of frames in the video  
        """
        
        n = len(self.frames)
        path = np.zeros((n,3))

        curr_R , curr_t = np.identity(3) , np.array([[0],[0],[0]])
        curr_pose = None
        path[0] = curr_t[:,0]

        # get key points and their descriptors for all frames
        kps,des = self.get_kps_data()

        for i in range(len(self.frames)-1):

            # try:
                # get matches
                l_pts_2d, r_pts_2d = self.get_matches(kps[i],des[i],kps[i+1],des[i+1],num_matches=100)

                # get fundamenal matrix ransac
                # F,l_pts_2d_inliers,r_pts_2d_inliers = self.get_fundamental_matrix_ransac(l_pts_2d,r_pts_2d,K,max_iters=500,error_thresh=0.1)

                # get fundamental matrix
                F = self.compute_fundamental_matrix_normalized(l_pts_2d,r_pts_2d)
                # check inliers
                num_inliers,l_pts_2d_inliers,r_pts_2d_inliers = self.compute_inliers(l_pts_2d,r_pts_2d,F,thresh=0.01)
                # compute F again using inliers only
                if num_inliers >= 8:
                    F = self.compute_fundamental_matrix_normalized(l_pts_2d_inliers,r_pts_2d_inliers)
                else:
                    l_pts_2d_inliers = l_pts_2d
                    r_pts_2d_inliers = r_pts_2d

                # get E from F enforcing constraints
                E = self.K.T @ F @ self.K
                U,S,V = np.linalg.svd(E)
                E = U @ np.diag([1,1,0]) @ V

                # decompose E into R and t and get the best out of the 4 solutions
                R,t = self.decompose_E(E,l_pts_2d_inliers,r_pts_2d_inliers)
                # _,R,t,_ = cv2.recoverPose(E,l_pts_2d.T,r_pts_2d.T,focal=self.focal_length,pp=self.pp)

                # get current pose
                t = self.get_scale(i+1) * t
                T = self.form_transformation_matrix(R,t) # 4x4

                if i == 0:
                    curr_pose = T 
                    path[i+1] = curr_pose[:3,3]
                else:
                    curr_pose = curr_pose @ np.linalg.inv(T)
                    path[i+1] =  curr_pose[:3,3]
            # except:
            #     path[i+1] = path[i]

        return path
        

if __name__=="__main__":
    frame_path = 'video_train'
    odemotryc = OdometryClass(frame_path)
    path = odemotryc.run()
    np.save('predictions.npy',path)

