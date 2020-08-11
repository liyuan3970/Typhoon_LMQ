;********************************************************
; Plot storm stracks from wrfout files.
;********************************************************
;
; JUN-18-2005 
; So-Young Ha (MMM/NCAR)
; SEP-01-2006
; Slightly modified by Mary Haley to add some extra comments.
; ===========================================

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRF_contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRFUserARW.ncl"


begin


f = addfile("/home/liyuan3970/WRF_Demo/demo4_run/geo_em.d01.nc","r")
hgt  = f->HGT_M
printVarSummary(hgt)
hgt_m = new((/1,87,99/), "float")
copy_VarCoords(hgt, hgt_m)
printVarSummary(hgt_m)
hgt_m = 500

end