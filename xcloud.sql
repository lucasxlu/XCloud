/*
Navicat MySQL Data Transfer

Source Server         : xcloud
Source Server Version : 50715
Source Host           : localhost:3306
Source Database       : xcloud

Target Server Type    : MYSQL
Target Server Version : 50715
File Encoding         : 65001

Date: 2019-03-21 20:06:43
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for api
-- ----------------------------
DROP TABLE IF EXISTS `api`;
CREATE TABLE `api` (
  `username` varchar(16) NOT NULL,
  `api_name` varchar(20) NOT NULL,
  `api_elapse` float(4,0) NOT NULL,
  `api_call_datetime` datetime NOT NULL ON UPDATE CURRENT_TIMESTAMP,
  `terminal_type` int(3) DEFAULT NULL,
  `img_path` varchar(100) DEFAULT NULL,
  `skin_disease` varchar(30) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of api
-- ----------------------------
INSERT INTO `api` VALUES ('LucasXU', 'cv/mcloud/skin', '0', '2019-03-21 17:45:05', '0', null, null);
INSERT INTO `api` VALUES ('LucasXU', 'cv/mcloud/skin', '0', '2019-02-06 17:45:34', '0', null, null);
INSERT INTO `api` VALUES ('LucasXU', 'cv/mcloud/skin', '0', '2019-03-14 17:45:58', '1', null, null);
INSERT INTO `api` VALUES ('BigBear', 'cv/fbp', '0', '0000-00-00 00:00:00', '1', null, null);
INSERT INTO `api` VALUES ('LucasXU', 'cv/mcloud/skin', '0', '0000-00-00 00:00:00', '0', 'http://localhost:8000/static/SkinUpload/009899HB.jpg', 'Acne_Keloidalis_Nuchae\n');
INSERT INTO `api` VALUES ('LucasXU', 'cv/mcloud/skin', '1', '2019-03-21 20:02:31', '0', 'http://localhost:8000/static/SkinUpload/009895HB.jpg', 'Acne_Keloidalis_Nuchae\n');

-- ----------------------------
-- Table structure for users
-- ----------------------------
DROP TABLE IF EXISTS `users`;
CREATE TABLE `users` (
  `username` varchar(16) NOT NULL,
  `register_datetime` datetime NOT NULL ON UPDATE CURRENT_TIMESTAMP,
  `register_type` int(11) NOT NULL,
  `user_organization` varchar(100) NOT NULL,
  `email` varchar(50) NOT NULL,
  `userkey` varchar(20) NOT NULL,
  `password` varchar(12) NOT NULL,
  PRIMARY KEY (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of users
-- ----------------------------
INSERT INTO `users` VALUES ('SmallBear', '0000-00-00 00:00:00', '0', 'Tsinghua University', 'smallbear@thu.edu.cn', 'smallbear', '66666a');
