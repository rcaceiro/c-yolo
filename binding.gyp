{
 "variables":{
  "with_opencv%":"<!(node ./util/has_lib.js opencv)",
  "with_cuda%":"<!(node ./util/has_lib.js cuda)"
 },
 "targets":[
  {
   "target_name":"nodeyolojs",
   "sources":[
    "module.cpp"
   ],
   "libraries":[
    "-lm",
    "-pthread",
    "-lstdc++"
   ],
   "include_dirs":[
    "./src",
    "./"
   ],
   "cflags":[
    "-Wall",
    "-Wfatal-errors",
    "-fPIC",
    "-Ofast"
   ],
   "conditions":[
    [
	'with_opencv=="true"',
	{
	 "defines":[
	  "OPENCV"
	 ],
	 "libraries":[
	  "-lopencv_core",
	  "-lopencv_highgui"
	 ]
	}
    ],
    [
	'with_cuda=="true"',
	{
	 "defines":[
	  "GPU"
	 ],
	 "libraries":[
	  "-L/usr/local/cuda/lib",
	  "-lcuda",
	  "-lcudart",
	  "-lcublas",
	  "-lcurand"
	 ],
	 "include_dirs":[
	  "./src",
	  "/usr/local/cuda/include"
	 ]
	}
    ]
   ]
  }
 ]
}