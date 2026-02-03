
import rospy
import actionlib
import subprocess
from pal_interaction_msgs.msg import TtsAction, TtsGoal, TtsResult, TtsFeedback

class TtsActionServer:
         def __init__(self):
            self.server = actionlib.SimpleActionServer('/tts', TtsAction, self.execute, False)
            self.server.register_preempt_callback(self.preempt_callback)
            self.process = None
            self.server.start()

         def execute(self, goal):
            rospy.loginfo("Received goal: %s", goal.rawtext.text)
            try:
               # Start the espeak-ng process with the given language and text
               language_dict = {'en_GB':'en'}
               self.process = subprocess.Popen(['espeak-ng', '-v', language_dict.get(goal.rawtext.lang_id), goal.rawtext.text])
               self.process.wait()
               result = TtsResult()
               result.text = goal.rawtext.text
               result.msg = "Speech synthesis completed successfully."
               self.server.set_succeeded(result)
            except Exception as e:
               rospy.logerr("Error during speech synthesis: %s", str(e))
               result = TtsResult()
               result.text = ""
               result.msg = "Error during speech synthesis."
               self.server.set_aborted(result)

         def preempt_callback(self):
            rospy.loginfo("Preempt request received.")
            if self.process:
               self.process.terminate()
               self.process = None
            result = TtsResult()
            result.text = ""
            result.msg = "Speech synthesis preempted."
            self.server.set_preempted(result)

if __name__ == '__main__':
         rospy.init_node('tts_action_server')
         server = TtsActionServer()
         rospy.spin()
