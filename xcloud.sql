/*
Navicat MySQL Data Transfer

Source Server         : XCloud
Source Server Version : 50714
Source Host           : localhost:3306
Source Database       : xcloud

Target Server Type    : MYSQL
Target Server Version : 50714
File Encoding         : 65001

Date: 2019-03-21 16:55:52
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for api
-- ----------------------------
DROP TABLE IF EXISTS `api`;
CREATE TABLE `api` (
  `username` varchar(16) NOT NULL,
  `api_name` varchar(20) NOT NULL,
  `api_elapse` int(10) NOT NULL,
  `api_call_datetime` datetime NOT NULL ON UPDATE CURRENT_TIMESTAMP,
  `terminal_type` int(3) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of api
-- ----------------------------

-- ----------------------------
-- Table structure for username
-- ----------------------------
DROP TABLE IF EXISTS `username`;
CREATE TABLE `username` (
  `username` varchar(16) NOT NULL,
  `register_datetime` datetime NOT NULL ON UPDATE CURRENT_TIMESTAMP,
  `register_type` int(11) NOT NULL,
  `user_organization` varchar(100) NOT NULL,
  `email` varchar(50) NOT NULL,
  `key` varchar(20) NOT NULL,
  `password` varchar(12) NOT NULL,
  PRIMARY KEY (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of username
-- ----------------------------
