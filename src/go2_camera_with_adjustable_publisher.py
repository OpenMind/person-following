import base64
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from unitree_api.msg import Request, RequestHeader, RequestIdentity, Response


class Go2CameraNode(Node):
    """ROS 2 node that requests JPEG frames from Unitree videohub and publishes /camera/image_raw."""

    def __init__(self):
        super().__init__("go2_camera_node")

        # Unitree videohub API
        self.VIDEO_API_VERSION = "1.0.0.1"
        self.VIDEO_API_ID_GETIMAGESAMPLE = 1001

        # Initialize CV Bridge for image conversion
        self.bridge = CvBridge()

        # Configurable output resolution for /camera/image_raw topic
        self.output_width = 640  # 640 for 640x480 or 1920 for 1920x1080
        self.output_height = 480  # 480 for 640x480 or 1080 for 1920x1080
        self.resize_enabled = True  # False to publish original resolution

        # Request throttling (prevents piling up requests if responses lag)
        self.request_hz = 30.0
        self._awaiting_image = False

        # Stats (to rate-limit warnings)
        self.decode_ok_count = 0
        self.decode_fail_count = 0

        # Publisher for camera request
        self.camera_request_publisher = self.create_publisher(
            Request,
            "/api/videohub/request",
            10,
        )

        # Subscriber for camera response
        self.camera_response_subscription = self.create_subscription(
            Response,
            "/api/videohub/response",
            self.camera_response_callback,
            10,
        )

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.image_publisher = self.create_publisher(
            Image,
            "/camera/image_raw",
            image_qos,
        )

        # Timer to request camera images
        self.create_timer(1.0 / self.request_hz, self.request_camera_image)

        if self.resize_enabled:
            self.get_logger().info(
                f"Go2 Camera Node initialized - Output resolution: {self.output_width}x{self.output_height}"
            )
        else:
            self.get_logger().info(
                "Go2 Camera Node initialized - Publishing original resolution"
            )

    def set_output_resolution(self, width: int, height: int) -> None:
        """Dynamically change the output resolution."""
        self.output_width = int(width)
        self.output_height = int(height)
        self.resize_enabled = True
        self.get_logger().info(f"Output resolution changed to: {width}x{height}")

    def disable_resize(self) -> None:
        """Disable resizing and publish original camera resolution."""
        self.resize_enabled = False
        self.get_logger().info("Resize disabled - publishing original resolution")

    def request_camera_image(self) -> None:
        """Send a videohub request for the next JPEG image sample."""
        if self._awaiting_image:
            # If the response stream lags, avoid stacking requests.
            return

        self._awaiting_image = True

        request_msg = Request()
        request_msg.header = RequestHeader()
        request_msg.header.identity = RequestIdentity()
        request_msg.header.identity.api_id = self.VIDEO_API_ID_GETIMAGESAMPLE
        request_msg.parameter = ""

        self.camera_request_publisher.publish(request_msg)
        self.get_logger().debug("Requested camera image sample")

    def camera_response_callback(self, msg: Response) -> None:
        """Handle videohub responses."""
        if msg.header.identity.api_id != self.VIDEO_API_ID_GETIMAGESAMPLE:
            self.get_logger().warning(
                f"Received unknown response with API ID: {msg.header.identity.api_id}"
            )
            return

        try:
            self.process_and_publish_image(msg.binary)
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
        finally:
            # Always release the outstanding flag, even if decode fails.
            self._awaiting_image = False

    @staticmethod
    def _extract_jpeg_payload(raw: bytes) -> Optional[bytes]:
        """
        Extract the JPEG payload from a raw byte buffer.

        Why:
        - Some frames may contain extra bytes before the JPEG SOI marker (FFD8).
        - Some frames may be truncated (missing EOI marker FFD9).

        We only attempt to decode when we can locate a full SOI..EOI range.
        """
        soi = raw.find(b"\xff\xd8")
        if soi < 0:
            return None

        eoi = raw.rfind(b"\xff\xd9")
        if eoi < 0 or eoi <= soi:
            return None

        return raw[soi : eoi + 2]

    @staticmethod
    def _maybe_base64_to_bytes(raw: bytes) -> Optional[bytes]:
        """
        Best-effort base64 decode for rare cases where payload is ascii/base64.
        """
        # Fast check: most base64 JPEGs start with '/9j/'
        if b"/9j/" not in raw[:64]:
            return None
        try:
            s = raw.decode("ascii", errors="ignore").strip()
            if "base64," in s:
                s = s.split("base64,", 1)[1]
            if len(s) < 100:
                return None
            return base64.b64decode(s, validate=False)
        except Exception:
            return None

    def process_and_publish_image(self, binary_data) -> None:
        """
        Process the received binary image data and display it using OpenCV imshow.

        Parameters
        ----------
        binary_data : list
            List of signed integers representing the JPEG image data
        """
        if not binary_data:
            return

        # int8[] -> bytes (preserves original byte pattern; no slow Python loops)
        raw = np.asarray(binary_data, dtype=np.int8).tobytes()

        jpeg = self._extract_jpeg_payload(raw)
        if jpeg is None:
            # Try base64 fallback (rare)
            jpeg = self._maybe_base64_to_bytes(raw)

        if jpeg is None:
            self.decode_fail_count += 1
            if self.decode_fail_count % 30 == 1:
                self.get_logger().warning(
                    f"Skipping frame: incomplete JPEG payload (fail_count={self.decode_fail_count})"
                )
            return

        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            self.decode_fail_count += 1
            if self.decode_fail_count % 30 == 1:
                self.get_logger().warning(
                    f"cv2.imdecode returned None (fail_count={self.decode_fail_count})"
                )
            return

        self.decode_ok_count += 1

        # Resize if enabled
        if self.resize_enabled and (
            image.shape[1] != self.output_width or image.shape[0] != self.output_height
        ):
            interpolation = (
                cv2.INTER_AREA
                if (self.output_width < image.shape[1])
                else cv2.INTER_LINEAR
            )
            image = cv2.resize(
                image,
                (self.output_width, self.output_height),
                interpolation=interpolation,
            )

        ros_image = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        ros_image.header.stamp = self.get_clock().now().to_msg()
        ros_image.header.frame_id = "camera_frame"
        self.image_publisher.publish(ros_image)

    def destroy_node(self):
        """`
        Clean up resources when the node is destroyed.
        """
        super().destroy_node()


def main(args=None):
    """Main function for Go2CameraNode."""
    rclpy.init(args=args)
    node = Go2CameraNode()

    # Example: You can change resolution during runtime
    # Uncomment these lines to test dynamic resolution changes:

    # For 640x480 output:
    # node.set_output_resolution(640, 480)

    # For 1920x1080 output:
    # node.set_output_resolution(1920, 1080)

    # For original resolution (no resizing):
    # node.disable_resize()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
