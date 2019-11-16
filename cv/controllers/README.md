# Controllers
## Introduction
According to [MVC](https://en.wikipedia.org/wiki/Model%E2%80%93view%E2%80%93controller) design pattern, model, controller and view should be 
organized separately. Hence this module is designed for holding ``controllers``.

The role of ``controller`` is to communicate with HTTP request directly, call
 corresponding machine learning models, and return inference results in JSON 
 format.
 

## Status Code
| code | desc |
| :---: | :---: |
| 0 | Success | 
| 1 | Invalid Image | 
| 2 | Invalid HTTP Method | 
| N | Self-defined | 
