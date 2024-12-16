import numpy as np
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
import pybullet as p


class ObS12Rw5(BaseRLAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=np.array([[0, 0, 0]]),
                 initial_rpys=np.array([[0, 0, 0]]),
                 target_xyzs=np.array([0, 0, 1]),
                 target_rpys=np.array([[0, 0, np.random.uniform(0, 1.5 + 1e-10, 1)[0]]]),
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.INIT_XYZS = initial_xyzs
        self.TARGET_POS = target_xyzs
        self.TARGET_ORIENTATION = target_rpys
        self.EPISODE_LEN_SEC = 5
        self.LOG_ANGULAR_VELOCITY = np.zeros((1, 3))
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    ################################################################################

    def _target_error(self, state):
        return (np.linalg.norm(self.TARGET_POS - state[0:3]) +
                np.linalg.norm(self.TARGET_ORIENTATION - state[7:10]))

    def _is_away_from_exploration_area(self, state):
        return (np.linalg.norm(state[0:2] - self.TARGET_POS[0:2]) >
                np.linalg.norm(self.INIT_XYZS[0][0:2] - self.TARGET_POS[0:2]) + 0.0125 or
                state[2] > self.TARGET_POS[2] + 0.0125)

    def _is_closed(self, state):
        return np.linalg.norm(state[0:3] - self.TARGET_POS[0:3]) < 0.02

    def _performance(self, state):
        if self._is_closed(state) and state[7]**2 + state[8]**2 < 0.001:
            return 2

        return -(state[7]**2 + state[8]**2)

    def _get_previous_current_we(self, current_state):
        if np.shape(self.LOG_ANGULAR_VELOCITY)[0] > 2:
            self.LOG_ANGULAR_VELOCITY = np.delete(self.LOG_ANGULAR_VELOCITY, 0, axis=0)

        return np.vstack((self.LOG_ANGULAR_VELOCITY, current_state[13:16]))

    def _get_we_differences(self, state):
        log = self._get_previous_current_we(state)
        differences = {
            'roll': log[0][0] - log[1][0],
            'pitch': log[0][1] - log[1][1],
            'yaw': log[0][2] - log[1][2],
        }
        return differences

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        we_differences = self._get_we_differences(state)
        ret = (25 - 20 * self._target_error(state) -
               100 * (1 if self._is_away_from_exploration_area(state) else -0.2) +
               20 * self._performance(state) -
               18 * (we_differences['roll']**2 + we_differences['pitch']**2 + we_differences['yaw']**2))
        return ret

    ################################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS - state[0:3]) < .02 and state[7]**2 + state[8]**2 < 0.001:
            return True

        return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        if (np.linalg.norm(state[0:2] - self.TARGET_POS[0:2]) >
                np.linalg.norm(self.INIT_XYZS[0][0:2] - self.TARGET_POS[0:2]) + .025 or
                state[2] > self.TARGET_POS[2] + .025 or
                abs(state[7]) > .25 or abs(state[8]) > .25):
            return True

        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True

        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42}  # Calculated by the Deep Thought supercomputer in 7.5M years

    ################################################################################

    def _observationSpace(self):
        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([[lo, lo, 0, lo, lo, lo, lo, lo, lo, lo, lo, lo]])
        obs_upper_bound = np.array([[hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi]])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    ################################################################################

    def _computeObs(self):
        obs_12 = np.zeros((self.NUM_DRONES, 12))
        for i in range(self.NUM_DRONES):
            obs = self._getDroneStateVector(i)
            obs_12[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12, )
        ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
        return ret

    ###############################################################################

    def reset(self,
              seed: int = None,
              options: dict = None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict[..], optional
            Additinonal options, unused

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """
        p.resetSimulation(physicsClientId=self.CLIENT)
        self._housekeeping()
        self._updateAndStoreKinematicInformation()
        self.INIT_XYZS = np.array([[
            np.random.uniform(-2, 2 + 1e-10, 1)[0],
            np.random.uniform(-2, 2 + 1e-10, 1)[0],
            np.random.uniform(0, 2 + 1e-10, 1)[0]]])
        self.INIT_RPYS = np.array([[
            np.random.uniform(-0.2, 0.2 + 1e-10, 1)[0],
            np.random.uniform(-0.2, 0.2 + 1e-10, 1)[0],
            np.random.uniform(-1.5, 1.5 + 1e-10, 1)[0]
        ]])
        self.TARGET_ORIENTATION = np.array([[0, 0, np.random.uniform(-1.5, 1.5 + 1e-10, 1)[0]]])
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        return initial_obs, initial_info
