#!/usr/bin/env python3
import json
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

try:
    from robodost_msgs.msg import VisionUpdate
    _HAS_ROBODOST_MSGS = True
except ImportError:
    _HAS_ROBODOST_MSGS = False

from .pipeline import RoboSignFormerPipeline


class SignNode(Node):
    """
    RoboSignFormer Türk İşaret Dili Tanıma ROS2 Düğümü.
    Hem standart '/robodost/tts_speak' (std_msgs/String) hem de '/robodost/sign_update' yayınlar;
    böylece Piper TTS ve LLM ekibi hiçbir ek kurulum yapmadan algılanan işareti seslendirebilir.
    """

    def __init__(self):
        super().__init__("sign_node")

        self.declare_parameter("model_dir", "")
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("sign_topic", "/robodost/sign_update")
        self.declare_parameter("tts_topic", "/robodost/tts_speak")
        self.declare_parameter("debug_image_topic", "/robodost/sign_debug_image")
        self.declare_parameter("conf_threshold", 0.55)
        self.declare_parameter("mirror_camera", False)
        self.declare_parameter("show_debug_window", False)

        model_dir = self.get_parameter("model_dir").get_parameter_value().string_value
        camera_topic = self.get_parameter("camera_topic").get_parameter_value().string_value
        sign_topic = self.get_parameter("sign_topic").get_parameter_value().string_value
        tts_topic = self.get_parameter("tts_topic").get_parameter_value().string_value
        debug_topic = self.get_parameter("debug_image_topic").get_parameter_value().string_value
        conf_threshold = self.get_parameter("conf_threshold").get_parameter_value().double_value
        mirror_camera = self.get_parameter("mirror_camera").get_parameter_value().bool_value
        self.show_debug_window = self.get_parameter("show_debug_window").get_parameter_value().bool_value

        self.pipeline = RoboSignFormerPipeline(
            model_dir=model_dir,
            conf_threshold=conf_threshold,
            mirror_camera=mirror_camera,
        )
        self.bridge = CvBridge()

        self.create_subscription(Image, camera_topic, self.image_callback, qos_profile_sensor_data)
        self.tts_pub = self.create_publisher(String, tts_topic, 10)
        self.debug_pub = self.create_publisher(Image, debug_topic, 5)

        if _HAS_ROBODOST_MSGS:
            self.sign_pub = self.create_publisher(VisionUpdate, sign_topic, 10)
        else:
            self.sign_pub = self.create_publisher(String, sign_topic, 10)
            self.get_logger().warn("ℹ️ robodost_msgs kurulu değil; sign_update std_msgs/String JSON olarak yayınlanıyor.")

        self.get_logger().info("🤟 RoboSignFormer TİD ROS2 Düğümü aktif!")

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Görüntü dönüşüm hatası: {e}")
            return

        need_hud = self.show_debug_window or (self.debug_pub.get_subscription_count() > 0)
        kelime, skor, vis = self.pipeline.process_frame(frame, draw_hud=need_hud)

        if kelime:
            tts_msg = String()
            tts_msg.data = kelime.replace("_", " ")
            self.tts_pub.publish(tts_msg)

            if _HAS_ROBODOST_MSGS:
                u_msg = VisionUpdate()
                u_msg.id_adi = kelime
                u_msg.guven_skoru = int(skor)
                u_msg.nesne_bulundu = True
                self.sign_pub.publish(u_msg)
            else:
                s_msg = String()
                s_msg.data = json.dumps({"kelime": kelime, "guven_skoru": skor, "nesne_bulundu": True}, ensure_ascii=False)
                self.sign_pub.publish(s_msg)

            self.get_logger().info(f"🤟 İşaret Algılandı: {kelime} (%{skor})")

        if self.debug_pub.get_subscription_count() > 0:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(vis, encoding="bgr8"))

        if self.show_debug_window:
            cv2.imshow("RoboSignFormer - ROS2 Monitor", vis)
            cv2.waitKey(1)

    def destroy_node(self):
        self.pipeline.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SignNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 RoboSignFormer düğümü kapatıldı.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
