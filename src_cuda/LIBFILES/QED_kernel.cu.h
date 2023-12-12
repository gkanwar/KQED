/**
  Kernel when x and y are non-zero
**/
#include "KQED.h"

#include "chnr_dS.h"   // chnr_dS()
#include "chnr_dT.h"   // chnr_dT()
#include "chnr_dV.h"   // chnr_dV()

#include "getff-new.h" // set_invariants()

// hand-unrolled direct evaluation of the kernel
template<size_t(*F)(size_t)>
__device__
static void
CONSTRUCT_FULL_KERNEL( double *kp ,
                       double S,
		       struct STV x )
{
  const double *vv  = (const double*)x.Vv ;
  const double *sxv = (const double*)x.Sxv ;
  const double *syv = (const double*)x.Syv ;
  const double *txv = (const double*)x.Txv ;
  const double *tyv = (const double*)x.Tyv ;
  const double sxv0 = sxv[0]/4. ;
  const double sxv1 = sxv[1]/4. ;
  const double sxv2 = sxv[2]/4. ;
  const double sxv3 = sxv[3]/4. ;
  const double sxyv0 = (sxv[0]+syv[0])/4. ;
  const double sxyv1 = (sxv[1]+syv[1])/4. ;
  const double sxyv2 = (sxv[2]+syv[2])/4. ;
  const double sxyv3 = (sxv[3]+syv[3])/4. ;
  kp[F(0)] += S*(+vv[26]+vv[31]-vv[38]-vv[55]);
  kp[F(1)] += S*(+vv[10]+vv[15]+vv[34]+vv[51]-2*(txv[10]+tyv[10]+txv[15]+tyv[15]));
  kp[F(2)] += S*(-vv[6]-vv[18]+2*(txv[6]+tyv[6]));
  kp[F(3)] += S*(-vv[7]-vv[19]+2*(txv[7]+tyv[7]));
  kp[F(4)] += S*(-vv[10]-vv[15]+vv[34]+vv[51]-2*(txv[10]+txv[15])+2*(txv[10]+tyv[10]+txv[15]+tyv[15]));
  kp[F(5)] += S*(+vv[26]+vv[31]+vv[38]+vv[55]-2*(txv[26]+txv[31]));
  kp[F(6)] += S*(+vv[2]-vv[22]+vv[42]-vv[47]+vv[59]+vv[62]-(txv[47]+txv[62])-2*(txv[2]+tyv[2]+txv[42]+sxv2)+(txv[47]+tyv[47])-(txv[62]+tyv[62]));
  kp[F(7)] += S*(+vv[3]-vv[23]+vv[43]+vv[46]-vv[58]+vv[63]-(txv[43])-2*(txv[3]+tyv[3])-(txv[43]+tyv[43]+txv[58])+(txv[58]+tyv[58])-2*(txv[63]+sxv3));
  kp[F(8)] += S*(+vv[6]-vv[18]-2*(tyv[6]));
  kp[F(9)] += S*(-vv[2]-vv[22]+vv[42]+vv[47]+vv[59]-vv[62]+2*(txv[2]+tyv[2]+txv[22])-(2*txv[47]+tyv[47])+(2*txv[62]+tyv[62]));   
  kp[F(10)] += S*(-vv[26]+vv[31]-vv[38]-vv[55]+2*(txv[38]+sxv1));
  kp[F(11)] += S*(-vv[27]-vv[30]-vv[39]+vv[54]+(2*txv[39]+tyv[39]-tyv[54]));
  kp[F(12)] += S*(+vv[7]-vv[19]-2*(tyv[7]));
  kp[F(13)] += S*(-vv[3]-vv[23]-vv[43]+vv[46]+vv[58]+vv[63] +2*(txv[3]+tyv[3]+txv[23])+(2*txv[43]+tyv[43])-(2*txv[58]+tyv[58]));
  kp[F(14)] += S*(-vv[27]-vv[30]+vv[39]-vv[54]-(tyv[39])+(2*txv[54]+tyv[54]));
  kp[F(15)] += S*(+vv[26]-vv[31]-vv[38]-vv[55]+2*(txv[55]+sxv1));
  kp[F(16)] += S*(-vv[10]-vv[15]-vv[34]-vv[51]+2*(txv[10]+txv[15]));
  kp[F(17)] += S*(+vv[26]+vv[31]-vv[38]-vv[55]-2*(tyv[26]+tyv[31]));
  kp[F(18)] += S*(+vv[2]-vv[22]-vv[42]+vv[47]-vv[59]-vv[62]+2*(txv[22]+tyv[22]+txv[42]+sxv2)-tyv[47]+2*txv[62]+tyv[62]);
  kp[F(19)] += S*(+vv[3]-vv[23]-vv[43]-vv[46]+vv[58]-vv[63]
	    +2*(txv[23]+tyv[23])+(2*txv[43]+tyv[43])-(tyv[58])+2*(txv[63]+sxv3));
  kp[F(20)] += S*(-vv[26]-vv[31]-vv[38]-vv[55]+2*(txv[26]+tyv[26]+txv[31]+tyv[31]));
  kp[F(21)] += S*(-vv[10]-vv[15]+vv[34]+vv[51]);
  kp[F(22)] += S*(+vv[6]+vv[18]-2*(txv[18]+tyv[18]));
  kp[F(23)] += S*(+vv[7]+vv[19]-2*(txv[19]+tyv[19]));
  kp[F(24)] += S*(+vv[2]+vv[22]-vv[42]-vv[47]-vv[59]+vv[62]+
	    -2*(txv[2]+txv[22]+tyv[22])+(2*txv[47]+tyv[47])-(2*txv[62]+tyv[62]));
  kp[F(25)] += S*(+vv[6]-vv[18]+2*tyv[18]);
  kp[F(26)] += S*(+vv[10]-vv[15]+vv[34]+vv[51]-2*(txv[34]+sxv0));
  kp[F(27)] += S*(+vv[11]+vv[14]+vv[35]-vv[50]-(2*txv[35]+tyv[35])+(tyv[50]));
  kp[F(28)] += S*(+vv[3]+vv[23]+vv[43]-vv[46]-vv[58]-vv[63]-2*(txv[3])-(2*txv[43]+tyv[43])-2*(txv[23]+tyv[23])+(2*txv[58]+tyv[58]));
  kp[F(29)] += S*(+vv[7]-vv[19]+2*(tyv[19]));
  kp[F(30)] += S*(+vv[11]+vv[14]-vv[35]+vv[50]+(tyv[35])-(2*txv[50]+tyv[50]));
  kp[F(31)] += S*(-vv[10]+vv[15]+vv[34]+vv[51]-2*(txv[51]+sxv0));
  kp[F(32)] += S*(+vv[6]+vv[18]-2*(txv[6]));
  kp[F(33)] += S*(-vv[2]+vv[22]+vv[42]+vv[47]-vv[59]+vv[62]-2*(txv[22])-2*(txv[42]+tyv[42]+sxyv2)-(tyv[47])-(2*txv[62]+tyv[62]));
  kp[F(34)] += S*(+vv[26]-vv[31]-vv[38]+vv[55]+2*(tyv[38]-sxv1+sxyv1));
  kp[F(35)] += S*(+vv[27]+vv[30]-vv[39]-vv[54]+(tyv[39])+(tyv[54]));
  kp[F(36)] += S*(-vv[2]+vv[22]-vv[42]-vv[47]+vv[59]-vv[62]+2*(txv[2]+txv[42]+tyv[42]+sxyv2)+(tyv[47])+(2*txv[62]+tyv[62]));
  kp[F(37)] += S*(-vv[6]-vv[18]+2*txv[18]);
  kp[F(38)] += S*(-vv[10]+vv[15]+vv[34]-vv[51]-2*(tyv[34]-sxv0+sxyv0));
  kp[F(39)] += S*(-vv[11]-vv[14]+vv[35]+vv[50]-tyv[35]-tyv[50]);
  kp[F(40)] += S*(+vv[26]+vv[31]+vv[38]-vv[55]-2*(txv[38]+tyv[38]+sxyv1));
  kp[F(41)] += S*(-vv[10]-vv[15]-vv[34]+vv[51]+2*(txv[34]+tyv[34]+sxyv0));
  kp[F(42)] += S*(+vv[6]-vv[18]);
  kp[F(43)] += S*(+vv[7]-vv[19]);
  kp[F(44)] += S*(-vv[27]+vv[30]+vv[39]+vv[54]-(tyv[39])-(2*txv[54]+tyv[54]));
  kp[F(45)] += S*(+vv[11]-vv[14]-vv[35]-vv[50]+tyv[35]+2*txv[50]+tyv[50]);
  kp[F(46)] += S*(-vv[7]+vv[19]);
  kp[F(47)] += S*(+vv[6]-vv[18]);
  kp[F(48)] += S*(+vv[7]+vv[19]-2*txv[7]);
  kp[F(49)] += S*(-vv[3]+vv[23]+vv[43]-vv[46]+vv[58]+vv[63]-2*(txv[23]+txv[43]+txv[63])-tyv[43]-tyv[58]-2*(tyv[63]+sxyv3));
  kp[F(50)] += S*(+vv[27]+vv[30]-vv[39]-vv[54]+tyv[39]+tyv[54]);
  kp[F(51)] += S*(-vv[26]+vv[31]+vv[38]-vv[55]-2*(+sxv1-tyv[55]-sxyv1));
  kp[F(52)] += S*(-vv[3]+vv[23]-vv[43]+vv[46]-vv[58]-vv[63]+2*(txv[3]+txv[63]+tyv[63]+sxyv3)+(2*txv[43]+tyv[43])+(tyv[58]));
  kp[F(53)] += S*(-vv[7]-vv[19]+2*(txv[19]));
  kp[F(54)] += S*(-vv[11]-vv[14]+vv[35]+vv[50]-tyv[35]-tyv[50]);
  kp[F(55)] += S*(+vv[10]-vv[15]-vv[34]+vv[51]-2*(tyv[51]-sxv0+sxyv0));  
  kp[F(56)] += S*(+vv[27]-vv[30]+vv[39]+vv[54]-2*txv[39]-tyv[39]-tyv[54]);
  kp[F(57)] += S*(-vv[11]+vv[14]-vv[35]-vv[50]+2*txv[35]+tyv[35]+tyv[50]);
  kp[F(58)] += S*(+vv[7]-vv[19]);
  kp[F(59)] += S*(-vv[6]+vv[18]);
  kp[F(60)] += S*(+vv[26]+vv[31]-vv[38]+vv[55]-2*(txv[55]+tyv[55]+sxyv1));
  kp[F(61)] += S*(-vv[10]-vv[15]+vv[34]-vv[51]+2*(txv[51]+tyv[51]+sxyv0));
  kp[F(62)] += S*(+vv[6]-vv[18]);
  kp[F(63)] += S*(+vv[7]-vv[19]);
  kp[F(64)] += S*(-vv[25]+vv[37]+vv[47]-vv[59]);
  kp[F(65)] += S*(-vv[9]-vv[33]+2*(txv[9]+tyv[9]));
  kp[F(66)] += S*(+vv[5]+vv[15]+vv[17]+vv[51]-2*(txv[5]+tyv[5]+txv[15]+tyv[15]));
  kp[F(67)] += S*(-vv[11]-vv[35]+2*(txv[11]+tyv[11]));
  kp[F(68)] += S*(+vv[9]-vv[33]-2*tyv[9]);
  kp[F(69)] += S*(-vv[25]-vv[37]+vv[47]-vv[59]+2*(txv[25]+sxv2));
  kp[F(70)] += S*(-vv[1]+vv[21]+vv[31]-vv[41]+vv[55]-vv[61]+2*(txv[1]+tyv[1])-(2*txv[31]+tyv[31])+2*(txv[41])+(2*txv[61]+tyv[61]));
  kp[F(71)] += S*(-vv[27]-vv[39]-vv[45]+vv[57]+(2*txv[27]+tyv[27])-tyv[57]);
  kp[F(72)] += S*(-vv[5]-vv[15]+vv[17]+vv[51]+2*(tyv[5]+tyv[15]));
  kp[F(73)] += S*(+vv[1]+vv[21]-vv[31]-vv[41]+vv[55]+vv[61]-2*(txv[1]+tyv[1])-2*(txv[21]+sxv1)+(tyv[31])-(2*txv[61]+tyv[61]));
  kp[F(74)] += S*(+vv[25]+vv[37]+vv[47]+vv[59]-2*(txv[37]+txv[47]));
  kp[F(75)] += S*(+vv[3]+vv[23]+vv[29]-vv[43]-vv[53]+vv[63]-2*(txv[3]+tyv[3])-(2*txv[23]+tyv[23])+(tyv[53])-2*(txv[63]+sxv3));
  kp[F(76)] += S*(+vv[11]-vv[35]-2*tyv[11]);
  kp[F(77)] += S*(+vv[27]-vv[39]-vv[45]-vv[57]-(tyv[27])+(2*txv[57]+tyv[57]));
  kp[F(78)] += S*(-vv[3]-vv[23]+vv[29]-vv[43]+vv[53]+vv[63]+2*(txv[3]+tyv[3]+txv[23]+txv[43]-txv[53])+tyv[23]-tyv[53]);
  kp[F(79)] += S*(-vv[25]+vv[37]-vv[47]-vv[59]+2*(txv[59]+sxv2));
  kp[F(80)] += S*(+vv[9]+vv[33]-2*txv[9]);
  kp[F(81)] += S*(-vv[25]+vv[37]-vv[47]+vv[59]+2*(tyv[25]-sxv2+sxyv2));
  kp[F(82)] += S*(-vv[1]+vv[21]+vv[31]+vv[41]-vv[55]+vv[61]-2*(txv[21]+tyv[21]+sxyv1)-(tyv[31])-2*(txv[41])-(2*txv[61]+tyv[61]));
  kp[F(83)] += S*(-vv[27]+vv[39]+vv[45]-vv[57]+(tyv[27]+tyv[57]));
  kp[F(84)] += S*(+vv[25]+vv[37]+vv[47]-vv[59]-2*(txv[25]+tyv[25]+sxyv2));
  kp[F(85)] += S*(+vv[9]-vv[33]);
  kp[F(86)] += S*(-vv[5]-vv[17]-vv[15]+vv[51]+2*(txv[17]+tyv[17]+sxyv0));
  kp[F(87)] += S*(+vv[11]-vv[35]);
  kp[F(88)] += S*(-vv[1]-vv[21]-vv[31]+vv[41]+vv[55]-vv[61]+2*(txv[1])+2*(txv[21]+tyv[21]+sxyv1)+(tyv[31])+(2*txv[61]+tyv[61]));
  kp[F(89)] += S*(-vv[5]+vv[15]+vv[17]-vv[51]+2*(sxv0-tyv[17]-sxyv0));
  kp[F(90)] += S*(-vv[9]-vv[33]+2*txv[33]);
  kp[F(91)] += S*(-vv[7]-vv[13]+vv[19]+vv[49]-(tyv[19]+tyv[49]));
  kp[F(92)] += S*(+vv[27]-vv[39]+vv[45]+vv[57]-(tyv[27]+2*txv[57]+tyv[57]));
  kp[F(93)] += S*(-vv[11]+vv[35]);
  kp[F(94)] += S*(+vv[7]-vv[13]-vv[19]-vv[49]+tyv[19]+2*txv[49]+tyv[49]);
  kp[F(95)] += S*(+vv[9]-vv[33]);
  kp[F(96)] += S*(-vv[5]-vv[15]-vv[17]-vv[51]+2*(txv[5]+txv[15]));
  kp[F(97)] += S*(+vv[1]-vv[21]+vv[31]-vv[41]-vv[55]-vv[61]+2*(txv[21]+sxv1)-(tyv[31])+2*(txv[41]+tyv[41])+(2*txv[61]+tyv[61]));
  kp[F(98)] += S*(-vv[25]+vv[37]+vv[47]-vv[59]-2*(tyv[37]+tyv[47]));
  kp[F(99)] += S*(+vv[3]-vv[23]-vv[29]-vv[43]+vv[53]-vv[63]+(2*txv[23]+tyv[23])+2*(txv[43]+tyv[43])-(tyv[53])+2*(txv[63]+sxv3));
  kp[F(100)] += S*(+vv[1]-vv[21]-vv[31]+vv[41]-vv[55]+vv[61]-2*(txv[1])+(2*txv[31]+tyv[31])-2*(txv[41]+tyv[41])-(2*txv[61]+tyv[61]));
  kp[F(101)] += S*(+vv[5]-vv[15]+vv[17]+vv[51]-2*(txv[17]+sxv0));
  kp[F(102)] += S*(+vv[9]-vv[33]+2*tyv[33]);
  kp[F(103)] += S*(+vv[7]+vv[13]+vv[19]-vv[49]-2*txv[19]-tyv[19]+tyv[49]);
  kp[F(104)] += S*(-vv[25]-vv[37]-vv[47]-vv[59]+2*(txv[37]+txv[47]+tyv[37]+tyv[47]));
  kp[F(105)] += S*(+vv[9]+vv[33]-2*(txv[33]+tyv[33]));
  kp[F(106)] += S*(-vv[5]-vv[15]+vv[17]+vv[51]);
  kp[F(107)] += S*(+vv[11]+vv[35]-2*(txv[35]+tyv[35]));
  kp[F(108)] += S*(+vv[3]+vv[23]-vv[29]+vv[43]-vv[53]-vv[63]-2*(txv[3]+txv[23]+txv[43]-txv[53])-(tyv[23]+2*tyv[43]-tyv[53]));
  kp[F(109)] += S*(+vv[7]+vv[13]-vv[19]+vv[49]+tyv[19]-2*txv[49]-tyv[49]);
  kp[F(110)] += S*(+vv[11]-vv[35]+2*tyv[35]);
  kp[F(111)] += S*(-vv[5]+vv[15]+vv[17]+vv[51]-2*(txv[51]+sxv0));
  kp[F(112)] += S*(+vv[11]+vv[35]-2*(txv[11]));
  kp[F(113)] += S*(-vv[27]+vv[39]+vv[45]-vv[57]+tyv[27]+tyv[57]);
  kp[F(114)] += S*(-vv[3]+vv[23]-vv[29]+vv[43]+vv[53]+vv[63]-2*(txv[23]+txv[43]+txv[63])-( tyv[23]+tyv[53]+2*(tyv[63]+sxyv3) ));
  kp[F(115)] += S*(+vv[25]-vv[37]+vv[47]-vv[59]+2*(tyv[59]-sxv2+sxyv2));
  kp[F(116)] += S*(+vv[27]+vv[39]-vv[45]+vv[57]-(2*txv[27]+tyv[27]+tyv[57]));
  kp[F(117)] += S*(+vv[11]-vv[35]);
  kp[F(118)] += S*(-vv[7]+vv[13]-vv[19]-vv[49]+2*txv[19]+tyv[19]+tyv[49]);
  kp[F(119)] += S*(-vv[9]+vv[33]);
  kp[F(120)] += S*(-vv[3]-vv[23]+vv[29]+vv[43]-vv[53]-vv[63]+2*(txv[3]+txv[23]+txv[63])+tyv[23]+tyv[53]+2*(tyv[63]+sxyv3));
  kp[F(121)] += S*(-vv[7]-vv[13]+vv[19]+vv[49]-tyv[19]-tyv[49]);
  kp[F(122)] += S*(-vv[11]-vv[35]+2*(txv[35]));
  kp[F(123)] += S*(+vv[5]-vv[15]-vv[17]+vv[51]-2*(tyv[51]-sxv0+sxyv0));
  kp[F(124)] += S*(-vv[25]+vv[37]+vv[47]+vv[59]-2*(txv[59]+tyv[59]+sxyv2));
  kp[F(125)] += S*(+vv[9]-vv[33]);
  kp[F(126)] += S*(-vv[5]-vv[15]+vv[17]-vv[51]+2*(txv[51]+tyv[51]+sxyv0));
  kp[F(127)] += S*(+vv[11]-vv[35]);
  kp[F(128)] += S*(-vv[29]-vv[46]+vv[53]+vv[58]);
  kp[F(129)] += S*(-vv[13]-vv[49]+2*(txv[13]+tyv[13]));
  kp[F(130)] += S*(-vv[14]-vv[50]+2*(txv[14]+tyv[14]));
  kp[F(131)] += S*(+vv[5]+vv[10]+vv[17]+vv[34]-2*(txv[5]+tyv[5]+txv[10]+tyv[10]));
  kp[F(132)] += S*(+vv[13]-vv[49]-2*tyv[13]);
  kp[F(133)] += S*(-vv[29]-vv[46]-vv[53]+vv[58]+2*(txv[29]+sxv3));
  kp[F(134)] += S*(-vv[30]+vv[45]-vv[54]-vv[57]+2*txv[30]+tyv[30]-tyv[45]);
  kp[F(135)] += S*(-vv[1]+vv[21]+vv[26]+vv[38]-vv[41]-vv[61]+2*(txv[1]-txv[26]+txv[41]+txv[61])+2*tyv[1]-tyv[26]+tyv[41] );
  kp[F(136)] += S*(+vv[14]-vv[50]-2*tyv[14]);
  kp[F(137)] += S*(+vv[30]-vv[45]-vv[54]-vv[57]+2*txv[45]-tyv[30]+tyv[45]);
  kp[F(138)] += S*(-vv[29]-vv[46]+vv[53]-vv[58]+2*(txv[46]+sxv3));
  kp[F(139)] += S*(-vv[2]-vv[22]+vv[25]+vv[37]+vv[42]-vv[62]+2*(txv[2]+txv[22]-txv[37]+txv[62])+2*tyv[2]+tyv[22]-tyv[37]);
  kp[F(140)] += S*(-vv[5]-vv[10]+vv[17]+vv[34]+2*(tyv[5]+tyv[10]));
  kp[F(141)] += S*(+vv[1]+vv[21]-vv[26]+vv[38]+vv[41]-vv[61]
	       -2*(txv[1]+txv[21]+sxv1+txv[41]) -2*tyv[1]+tyv[26]-tyv[41]);
  kp[F(142)] += S*(+vv[2]+vv[22]+vv[25]-vv[37]+vv[42]-vv[62]
	       -2*(txv[2]+txv[22]+txv[42]+sxv2)-2*tyv[2]-tyv[22]+tyv[37]) ; 
  kp[F(143)] += S*(+vv[29]+vv[46]+vv[53]+vv[58]-2*(txv[53]+txv[58]));
  kp[F(144)] += S*(+vv[13]+vv[49]-2*(txv[13]));
  kp[F(145)] += S*(-vv[29]+vv[46]+vv[53]-vv[58]+2*(tyv[29]-sxv3+sxyv3));
  kp[F(146)] += S*(-vv[30]-vv[45]+vv[54]+vv[57]+tyv[30]+tyv[45]);
  kp[F(147)] += S*(-vv[1]+vv[21]+vv[26]-vv[38]+vv[41]+vv[61]-2*(txv[21]+txv[41]+txv[61]+sxyv1)-2*tyv[21]-tyv[26]-tyv[41]);
  kp[F(148)] += S*(+vv[29]-vv[46]+vv[53]+vv[58]-2*(txv[29]+tyv[29]+sxyv3));
  kp[F(149)] += S*(+vv[13]-vv[49]);
  kp[F(150)] += S*(+vv[14]-vv[50]);
  kp[F(151)] += S*(-vv[5]-vv[10]-vv[17]+vv[34]+2*(txv[17]+tyv[17]+sxyv0));
  kp[F(152)] += S*(+vv[30]+vv[45]-vv[54]+vv[57]-tyv[30]-2*txv[45]-tyv[45]);
  kp[F(153)] += S*(-vv[14]+vv[50]);
  kp[F(154)] += S*(+vv[13]-vv[49]);
  kp[F(155)] += S*(+vv[6]-vv[9]-vv[18]-vv[33]+2*txv[33]+tyv[18]+tyv[33]);
  kp[F(156)] += S*(-vv[1]-vv[21]-vv[26]+vv[38]-vv[41]+vv[61]+2*(txv[1]+txv[21]+txv[41]+sxyv1)+2*tyv[21]+tyv[26]+tyv[41]) ;
  kp[F(157)] += S*(-vv[5]+vv[10]+vv[17]-vv[34]-2*(tyv[17]-sxv0+sxyv0));
  kp[F(158)] += S*(-vv[6]-vv[9]+vv[18]+vv[33]-(tyv[18]+tyv[33]));
  kp[F(159)] += S*(-vv[13]-vv[49]+2*(txv[49]));
  kp[F(160)] += S*(+vv[14]+vv[50]-2*(txv[14]));
  kp[F(161)] += S*(-vv[30]-vv[45]+vv[54]+vv[57]+tyv[30]+tyv[45]);
  kp[F(162)] += S*(+vv[29]-vv[46]-vv[53]+vv[58]+2*(tyv[46]-sxv3+sxyv3));
  kp[F(163)] += S*(-vv[2]+vv[22]-vv[25]+vv[37]+vv[42]+vv[62]-2*(txv[22]+txv[42]+txv[62]+sxyv2)-tyv[22]-tyv[37]-2*tyv[42] ) ;
  kp[F(164)] += S*(+vv[30]+vv[45]+vv[54]-vv[57]-2*txv[30]-tyv[30]-tyv[45]);
  kp[F(165)] += S*(+vv[14]-vv[50]);
  kp[F(166)] += S*(-vv[13]+vv[49]);
  kp[F(167)] += S*(-vv[6]+vv[9]-vv[18]-vv[33]+2*txv[18]+tyv[18]+tyv[33]);
  kp[F(168)] += S*(-vv[29]+vv[46]+vv[53]+vv[58]-2*(txv[46]+tyv[46]+sxyv3));
  kp[F(169)] += S*(+vv[13]-vv[49]);
  kp[F(170)] += S*(+vv[14]-vv[50]);
  kp[F(171)] += S*(-vv[5]-vv[10]+vv[17]-vv[34]+2*(txv[34]+tyv[34]+sxyv0));
  kp[F(172)] += S*(-vv[2]-vv[22]+vv[25]-vv[37]-vv[42]+vv[62]+2*(txv[2]+txv[22]+txv[42]+sxyv2)+tyv[22]+tyv[37]+2*tyv[42] ) ;
  kp[F(173)] += S*(-vv[6]-vv[9]+vv[18]+vv[33]-(tyv[18]+tyv[33]));
  kp[F(174)] += S*(+vv[5]-vv[10]-vv[17]+vv[34]-2*(tyv[34]-sxv0+sxyv0));
  kp[F(175)] += S*(-vv[14]-vv[50]+2*(txv[50]));
  kp[F(176)] += S*(-vv[5]-vv[10]-vv[17]-vv[34]+2*(txv[5]+txv[10]));
  kp[F(177)] += S*(+vv[1]-vv[21]+vv[26]-vv[38]-vv[41]-vv[61]+2*(txv[21]+txv[41]+txv[61]+sxv1)-tyv[26]+2*tyv[61]+tyv[41] ) ;
  kp[F(178)] += S*(+vv[2]-vv[22]-vv[25]+vv[37]-vv[42]-vv[62]+2*(txv[22]+txv[42]+txv[62]+sxv2)+tyv[22]-tyv[37]+2*tyv[62] ) ;
  kp[F(179)] += S*(-vv[29]-vv[46]+vv[53]+vv[58]-2*(tyv[53]+tyv[58]));
  kp[F(180)] += S*(+vv[1]-vv[21]-vv[26]-vv[38]+vv[41]+vv[61]-2*(txv[1]-txv[26]+txv[41]+txv[61])+tyv[26]-tyv[41]-2*tyv[61]);
  kp[F(181)] += S*(+vv[5]-vv[10]+vv[17]+vv[34]-2*(txv[17]+sxv0));
  kp[F(182)] += S*(+vv[6]+vv[9]+vv[18]-vv[33]-(2*txv[18]+tyv[18]-tyv[33]));
  kp[F(183)] += S*(+vv[13]-vv[49]+2*tyv[49]);
  kp[F(184)] += S*(+vv[2]+vv[22]-vv[25]-vv[37]-vv[42]+vv[62]-2*(txv[2]+txv[22]-txv[37]+txv[62])-tyv[22]+tyv[37]-2*tyv[62] ) ;
  kp[F(185)] += S*(+vv[9]+vv[6]-vv[18]+vv[33]+tyv[18]-2*txv[33]-tyv[33]);
  kp[F(186)] += S*(-vv[5]+vv[10]+vv[17]+vv[34]-2*(txv[34]+sxv0));
  kp[F(187)] += S*(+vv[14]-vv[50]+2*tyv[50]);
  kp[F(188)] += S*(-vv[29]-vv[46]-vv[53]-vv[58]+2*(txv[53]+tyv[53]+txv[58]+tyv[58]));
  kp[F(189)] += S*(+vv[13]+vv[49]-2*(txv[49]+tyv[49]));
  kp[F(190)] += S*(+vv[14]+vv[50]-2*(txv[50]+tyv[50]));
  kp[F(191)] += S*(-vv[5]-vv[10]+vv[17]+vv[34]);
  kp[F(192)] += S*(+vv[24]-vv[36]);
  kp[F(193)] += S*(+vv[8]+vv[32]+vv[47]-vv[59]-2*(txv[8]+tyv[8]+sxyv2));
  kp[F(194)] += S*(-vv[4]-vv[16]-vv[31]+vv[55]+2*(txv[4]+tyv[4]+sxyv1));
  kp[F(195)] += S*(+vv[27]-vv[39]);
  kp[F(196)] += S*(-vv[8]+vv[32]-vv[47]+vv[59]-2*(txv[8]+sxv2-txv[8]-tyv[8]-sxyv2));
  kp[F(197)] += S*(+vv[24]+vv[36]-2*txv[24]);
  kp[F(198)] += S*(+vv[0]+vv[15]-vv[20]+vv[40]-vv[51]+vv[60]-2*(txv[0]+txv[40]+txv[60]+sxyv0)-2*tyv[0]-tyv[15]-tyv[60]) ;
  kp[F(199)] += S*(-vv[11]+vv[35]+vv[44]-vv[56]+tyv[11]+tyv[56]);
  kp[F(200)] += S*(+vv[4]-vv[16]+vv[31]-vv[55]-2*(tyv[4]-sxv1+sxyv1));
  kp[F(201)] += S*(-vv[0]-vv[15]-vv[20]+vv[40]+vv[51]-vv[60]+2*(txv[0]+txv[20]+txv[60]+sxyv0)+2*tyv[0]+tyv[15]+tyv[60]);
  kp[F(202)] += S*(-vv[24]-vv[36]+2*(txv[36])); 
  kp[F(203)] += S*(+vv[7]-vv[19]-vv[28]+vv[52]-tyv[7]-tyv[52]);
  kp[F(204)] += S*(-vv[27]+vv[39]);
  kp[F(205)] += S*(+vv[11]-vv[35]+vv[44]+vv[56]-tyv[11]-2*txv[56]-tyv[56]);
  kp[F(206)] += S*(-vv[7]+vv[19]-vv[28]-vv[52]+tyv[7]+2*txv[52]+tyv[52]);
  kp[F(207)] += S*(+vv[24]-vv[36]);
  kp[F(208)] += S*(-vv[8]-vv[32]+vv[47]-vv[59]+2*(txv[8]+sxv2));
  kp[F(209)] += S*(+vv[24]-vv[36]-2*tyv[24]);
  kp[F(210)] += S*(+vv[0]+vv[15]-vv[20]-vv[40]+vv[51]-vv[60]+2*(-txv[15]+txv[20]+txv[40]+txv[60])-tyv[15]+2*tyv[20]+tyv[60]);
  kp[F(211)] += S*(-vv[11]-vv[35]-vv[44]+vv[56]+2*txv[11]+tyv[11]-tyv[56]);
  kp[F(212)] += S*(-vv[24]-vv[36]+2*(txv[24]+tyv[24]));
  kp[F(213)] += S*(-vv[8]+vv[32]+vv[47]-vv[59]);
  kp[F(214)] += S*(+vv[4]+vv[16]+vv[31]+vv[55]-2*(txv[31]+tyv[31]+txv[16]+tyv[16]));
  kp[F(215)] += S*(-vv[27]-vv[39]+2*(txv[27]+tyv[27]));
  kp[F(216)] += S*(+vv[0]-vv[15]+vv[20]-vv[40]+vv[51]+vv[60]-2*(txv[0]+txv[20]+txv[60]+sxv0)+tyv[15]-2*tyv[20]-tyv[60]) ;
  kp[F(217)] += S*(+vv[4]-vv[16]-vv[31]+vv[55]+2*(tyv[16]+tyv[31]));
  kp[F(218)] += S*(+vv[8]+vv[32]+vv[47]+vv[59]-2*(txv[32]+txv[47]));
  kp[F(219)] += S*(+vv[3]+vv[12]+vv[23]-vv[43]-vv[48]+vv[63]-2*(txv[3]+txv[23]+txv[63]+sxv3)-tyv[3]-2*tyv[23]+tyv[48]);
  kp[F(220)] += S*(+vv[11]-vv[35]-vv[44]-vv[56]+2*txv[56]-tyv[11]+tyv[56]);
  kp[F(221)] += S*(+vv[27]-vv[39]-2*tyv[27]);
  kp[F(222)] += S*(-vv[3]+vv[12]-vv[23]-vv[43]+vv[48]+vv[63]+2*(txv[3]+txv[23]+txv[43]-txv[48])+tyv[3]+2*tyv[23]-tyv[48]);
  kp[F(223)] += S*(-vv[8]+vv[32]-vv[47]-vv[59]+2*(txv[59]+sxv2));
  kp[F(224)] += S*(+vv[4]+vv[16]-vv[31]+vv[55]-2*(txv[4]+sxv1));  
  kp[F(225)] += S*(-vv[0]-vv[15]+vv[20]+vv[40]-vv[51]+vv[60]-2*(-txv[15]+txv[20]+txv[40]+txv[60])+tyv[15]-2*tyv[40]-tyv[60]);
  kp[F(226)] += S*(+vv[24]-vv[36]+2*tyv[36]);
  kp[F(227)] += S*(+vv[7]+vv[19]+vv[28]-vv[52]-2*txv[7]-tyv[7]+tyv[52]);
  kp[F(228)] += S*(-vv[0]+vv[15]+vv[20]-vv[40]-vv[51]-vv[60]+2*(txv[0]+txv[40]+txv[60]+sxv0)-tyv[15]+2*tyv[40]+tyv[60]);
  kp[F(229)] += S*(-vv[4]-vv[16]-vv[31]-vv[55]+2*(txv[16]+txv[31]));
  kp[F(230)] += S*(-vv[8]+vv[32]+vv[47]-vv[59]-2*(tyv[32]+tyv[47]));
  kp[F(231)] += S*(-vv[3]-vv[12]+vv[23]-vv[43]+vv[48]-vv[63]+2*(txv[3]+txv[43]+txv[63]+sxv3)+tyv[3]+2*tyv[43]-tyv[48]);
  kp[F(232)] += S*(+vv[24]+vv[36]-2*(txv[36]+tyv[36]));
  kp[F(233)] += S*(-vv[8]-vv[32]-vv[47]-vv[59]+2*(txv[32]+tyv[32]+txv[47]+tyv[47]));
  kp[F(234)] += S*(+vv[4]-vv[16]-vv[31]+vv[55]);
  kp[F(235)] += S*(+vv[27]+vv[39]-2*(txv[39]+tyv[39]));
  kp[F(236)] += S*(-vv[7]+vv[19]+vv[28]+vv[52]+tyv[7]-2*txv[52]-tyv[52]);
  kp[F(237)] += S*(+vv[3]-vv[12]+vv[23]+vv[43]-vv[48]-vv[63]-2*(txv[3]+txv[23]+txv[43]-txv[48])-tyv[3]-2*tyv[43]+tyv[48]);
  kp[F(238)] += S*(+vv[27]-vv[39]+2*tyv[39]);
  kp[F(239)] += S*(+vv[4]-vv[16]+vv[31]+vv[55]-2*(txv[55]+sxv1));
  kp[F(240)] += S*(+vv[27]-vv[39]);
  kp[F(241)] += S*(+vv[11]+vv[35]-vv[44]+vv[56]-2*txv[11]-tyv[11]-tyv[56]);
  kp[F(242)] += S*(-vv[7]-vv[19]+vv[28]-vv[52]+2*txv[7]+tyv[7]+tyv[52]);
  kp[F(243)] += S*(-vv[24]+vv[36]);
  kp[F(244)] += S*(-vv[11]+vv[35]+vv[44]-vv[56]+tyv[11]+tyv[56]);
  kp[F(245)] += S*(+vv[27]+vv[39]-2*txv[27]);
  kp[F(246)] += S*(+vv[3]-vv[12]-vv[23]+vv[43]+vv[48]+vv[63]-2*(txv[3]+txv[43]+txv[63]+sxyv3)-tyv[3]-tyv[48]-2*tyv[63]);
  kp[F(247)] += S*(+vv[8]-vv[32]+vv[47]-vv[59]+2*(tyv[59]-sxv2+sxyv2));
  kp[F(248)] += S*(+vv[7]-vv[19]-vv[28]+vv[52]-tyv[7]-tyv[52]);  
  kp[F(249)] += S*(-vv[3]+vv[12]-vv[23]+vv[43]-vv[48]-vv[63]+2*(txv[3]+txv[23]+txv[63]+sxyv3)+tyv[3]+tyv[48]+2*tyv[63]);
  kp[F(250)] += S*(-vv[27]-vv[39]+2*txv[39]);
  kp[F(251)] += S*(-vv[4]+vv[16]-vv[31]+vv[55]-2*(tyv[55]-sxv1+sxyv1));
  kp[F(252)] += S*(+vv[24]-vv[36]);
  kp[F(253)] += S*(-vv[8]+vv[32]+vv[47]+vv[59]-2*(txv[59]+tyv[59]+sxyv2));
  kp[F(254)] += S*(+vv[4]-vv[16]-vv[31]-vv[55]+2*(txv[55]+tyv[55]+sxyv1));
  kp[F(255)] += S*(+vv[27]-vv[39]);
  kp[F(256)] += S*(+vv[28]-vv[52]);
  kp[F(257)] += S*(+vv[12]-vv[46]+vv[48]+vv[58]-2*(txv[12]+tyv[12]+sxyv3));
  kp[F(258)] += S*(+vv[30]-vv[54]);
  kp[F(259)] += S*(-vv[4]-vv[16]-vv[26]+vv[38]+2*(txv[4]+tyv[4]+sxyv1));
  kp[F(260)] += S*(-vv[12]+vv[46]+vv[48]-vv[58]+2*(tyv[12]-sxv3+sxyv3));
  kp[F(261)] += S*(+vv[28]+vv[52]-2*txv[28]);
  kp[F(262)] += S*(-vv[14]-vv[44]+vv[50]+vv[56]+tyv[14]+tyv[44]); 
  kp[F(263)] += S*(+vv[0]+vv[10]-vv[20]-vv[34]+vv[40]+vv[60]-2*(txv[0]+txv[40]+txv[60]+sxyv0)-2*tyv[0]-tyv[10]-tyv[40]);
  kp[F(264)] += S*(-vv[30]+vv[54]);
  kp[F(265)] += S*(+vv[14]+vv[44]-vv[50]+vv[56]-tyv[14]-2*txv[44]-tyv[44]);
  kp[F(266)] += S*(+vv[28]-vv[52]);
  kp[F(267)] += S*(-vv[6]+vv[18]-vv[24]-vv[36]+tyv[6]+2*txv[36]+tyv[36]);
  kp[F(268)] += S*(+vv[4]-vv[16]+vv[26]-vv[38]-2*(tyv[4]-sxv1+sxyv1)); 
  kp[F(269)] += S*(-vv[0]-vv[10]-vv[20]+vv[34]-vv[40]+vv[60]+2*(txv[0]+txv[20]+txv[40]+sxyv0)+2*tyv[0]+tyv[10]+tyv[40]);
  kp[F(270)] += S*(+vv[6]-vv[18]-vv[24]+vv[36]-tyv[6]-tyv[36]);
  kp[F(271)] += S*(-vv[28]-vv[52]+2*(txv[52]));
  kp[F(272)] += S*(-vv[12]-vv[46]-vv[48]+vv[58]+2*(txv[12]+sxv3));
  kp[F(273)] += S*(+vv[28]-vv[52]-2*tyv[28]);
  kp[F(274)] += S*(-vv[14]+vv[44]-vv[50]-vv[56]+2*txv[14]+tyv[14]-tyv[44]);
  kp[F(275)] += S*(+vv[0]+vv[10]-vv[20]+vv[34]-vv[40]-vv[60]+2*(-txv[10]+txv[20]+txv[40]+txv[60])-tyv[10]+2*tyv[20]+tyv[40]);
  kp[F(276)] += S*(-vv[28]-vv[52]+2*(txv[28]+tyv[28]));
  kp[F(277)] += S*(-vv[12]-vv[46]+vv[48]+vv[58]);
  kp[F(278)] += S*(-vv[30]-vv[54]+2*(txv[30]+tyv[30]));
  kp[F(279)] += S*(+vv[26]+vv[38]+vv[4]+vv[16]-2*(txv[16]+txv[26]+tyv[16]+tyv[26]));
  kp[F(280)] += S*(+vv[14]-vv[44]-vv[50]-vv[56]-tyv[14]+2*txv[44]+tyv[44]);
  kp[F(281)] += S*(+vv[30]-vv[54]-2*tyv[30]);
  kp[F(282)] += S*(-vv[12]-vv[46]+vv[48]-vv[58]+2*(txv[46]+sxv3));
  kp[F(283)] += S*(-vv[2]+vv[8]-vv[22]+vv[32]+vv[42]-vv[62]+2*(txv[2]+txv[22]-txv[32]+txv[62])+tyv[2]+2*tyv[22]-tyv[32]);
  kp[F(284)] += S*(+vv[0]-vv[10]+vv[20]+vv[34]+vv[40]-vv[60]-2*(txv[0]+txv[20]+txv[40]+sxv0)+tyv[10]-2*tyv[20]-tyv[40]);
  kp[F(285)] += S*(+vv[4]-vv[16]-vv[26]+vv[38]+2*(tyv[16]+tyv[26]));  
  kp[F(286)] += S*(+vv[2]+vv[8]+vv[22]-vv[32]+vv[42]-vv[62]-2*(txv[2]+txv[22]+txv[42]+sxv2)-tyv[2]-2*tyv[22]+tyv[32]);
  kp[F(287)] += S*(+vv[12]+vv[46]+vv[48]+vv[58]-2*(txv[48]+txv[58]));
  kp[F(288)] += S*(+vv[30]-vv[54]);
  kp[F(289)] += S*(+vv[14]+vv[44]+vv[50]-vv[56]-2*txv[14]-tyv[14]-tyv[44]);
  kp[F(290)] += S*(-vv[28]+vv[52]);
  kp[F(291)] += S*(-vv[6]-vv[18]+vv[24]-vv[36]+2*txv[6]+tyv[6]+tyv[36]);
  kp[F(292)] += S*(-vv[14]-vv[44]+vv[50]+vv[56]+tyv[14]+tyv[44]);
  kp[F(293)] += S*(+vv[30]+vv[54]-2*txv[30]);
  kp[F(294)] += S*(+vv[12]-vv[46]-vv[48]+vv[58]+2*(tyv[46]-sxv3+sxyv3));
  kp[F(295)] += S*(+vv[2]-vv[8]-vv[22]+vv[32]+vv[42]+vv[62]-2*(txv[2]+txv[42]+txv[62]+sxyv2)-tyv[2]-tyv[32]-2*tyv[42]);
  kp[F(296)] += S*(+vv[28]-vv[52]);
  kp[F(297)] += S*(-vv[12]+vv[46]+vv[48]+vv[58]-2*(txv[46]+tyv[46]+sxyv3));
  kp[F(298)] += S*(+vv[30]-vv[54]);
  kp[F(299)] += S*(+vv[4]-vv[16]-vv[26]-vv[38]+2*(txv[38]+tyv[38]+sxyv1));
  kp[F(300)] += S*(+vv[6]-vv[18]-vv[24]+vv[36]-tyv[6]-tyv[36]);
  kp[F(301)] += S*(-vv[2]+vv[8]-vv[22]-vv[32]-vv[42]+vv[62]+2*(txv[2]+txv[22]+txv[42]+sxyv2)+tyv[2]+tyv[32]+2*tyv[42]);
  kp[F(302)] += S*(-vv[4]+vv[16]-vv[26]+vv[38]-2*(tyv[38]-sxv1+sxyv1));
  kp[F(303)] += S*(-vv[30]-vv[54]+2*(txv[54]));
  kp[F(304)] += S*(+vv[4]+vv[16]-vv[26]+vv[38]-2*(txv[4]+sxv1));
  kp[F(305)] += S*(-vv[0]-vv[10]+vv[20]-vv[34]+vv[40]+vv[60]-2*(-txv[10]+txv[20]+txv[40]+txv[60])+tyv[10]-tyv[40]-2*tyv[60]);
  kp[F(306)] += S*(+vv[6]+vv[18]+vv[24]-vv[36]-2*txv[6]-tyv[6]+tyv[36]);
  kp[F(307)] += S*(+vv[28]-vv[52]+2*tyv[52]);
  kp[F(308)] += S*(-vv[0]+vv[10]+vv[20]-vv[34]-vv[40]-vv[60]+2*(txv[0]+txv[40]+txv[60]+sxv0)-tyv[10]+tyv[40]+2*tyv[60]);
  kp[F(309)] += S*(-vv[4]-vv[16]-vv[26]-vv[38]+2*(txv[16]+txv[26]));
  kp[F(310)] += S*(-vv[2]-vv[8]+vv[22]+vv[32]-vv[42]-vv[62]+2*(txv[2]+txv[42]+txv[62]+sxv2)+tyv[2]-tyv[32]+2*tyv[62]);
  kp[F(311)] += S*(-vv[12]-vv[46]+vv[48]+vv[58]-2*(tyv[48]+tyv[58]));
  kp[F(312)] += S*(-vv[6]+vv[18]+vv[24]+vv[36]+tyv[6]-2*txv[36]-tyv[36]);
  kp[F(313)] += S*(+vv[2]-vv[8]+vv[22]-vv[32]-vv[42]+vv[62]-2*(txv[2]+txv[22]-txv[32]+txv[62])-tyv[2]+tyv[32]-2*tyv[62]);
  kp[F(314)] += S*(+vv[4]-vv[16]+vv[26]+vv[38]-2*(txv[38]+sxv1));
  kp[F(315)] += S*(+vv[30]-vv[54]+2*tyv[54]);
  kp[F(316)] += S*(+vv[28]+vv[52]-2*(txv[52]+tyv[52]));
  kp[F(317)] += S*(-vv[12]-vv[48]-vv[46]-vv[58]+2*(txv[48]+tyv[48]+txv[58]+tyv[58]));
  kp[F(318)] += S*(+vv[30]+vv[54]-2*(txv[54]+tyv[54]));
  kp[F(319)] += S*(+vv[4]-vv[16]-vv[26]+vv[38]);
  kp[F(320)] += S*(+vv[44]-vv[56]);
  kp[F(321)] += S*(+vv[45]-vv[57]);
  kp[F(322)] += S*(+vv[12]+vv[48]-vv[29]+vv[53]-2*(txv[12]+tyv[12]+sxyv3));
  kp[F(323)] += S*(-vv[8]+vv[25]-vv[32]-vv[37]+2*(txv[8]+tyv[8]+sxyv2));
  kp[F(324)] += S*(-vv[45]+vv[57]);
  kp[F(325)] += S*(+vv[44]-vv[56]);
  kp[F(326)] += S*(+vv[13]+vv[28]-vv[49]+vv[52]-tyv[13]-2*txv[28]-tyv[28]);
  kp[F(327)] += S*(-vv[9]-vv[24]+vv[33]-vv[36]+tyv[9]+2*txv[24]+tyv[24]);
  kp[F(328)] += S*(-vv[12]+vv[48]+vv[29]-vv[53]+2*(tyv[12]-sxv3+sxyv3));
  kp[F(329)] += S*(-vv[13]-vv[28]+vv[49]+vv[52]+tyv[13]+tyv[28]);
  kp[F(330)] += S*(+vv[44]+vv[56]-2*txv[44]);
  kp[F(331)] += S*(+vv[0]+vv[5]-vv[17]+vv[20]-vv[40]+vv[60]-2*(txv[0]+txv[20]+txv[60]+sxyv0)-2*tyv[0]-tyv[5]-tyv[20]);
  kp[F(332)] += S*(+vv[8]-vv[25]-vv[32]+vv[37]-2*(tyv[8]-sxv2+sxyv2));
  kp[F(333)] += S*(+vv[9]+vv[24]-vv[33]-vv[36]-tyv[9]-tyv[24]);
  kp[F(334)] += S*(-vv[0]-vv[5]+vv[17]-vv[20]-vv[40]+vv[60]+2*(txv[0]+txv[20]+txv[40]+sxyv0)+2*tyv[0]+tyv[5]+tyv[20]);
  kp[F(335)] += S*(-vv[44]-vv[56]+2*txv[56]);
  kp[F(336)] += S*(+vv[45]-vv[57]);
  kp[F(337)] += S*(-vv[44]+vv[56]);
  kp[F(338)] += S*(+vv[13]+vv[28]+vv[49]-vv[52]-2*txv[13]-tyv[13]-tyv[28]);
  kp[F(339)] += S*(-vv[9]-vv[24]-vv[33]+vv[36]+2*txv[9]+tyv[9]+tyv[24]);
  kp[F(340)] += S*(+vv[44]-vv[56]);
  kp[F(341)] += S*(+vv[45]-vv[57]);
  kp[F(342)] += S*(-vv[12]+vv[29]+vv[48]+vv[53]-2*(txv[29]+tyv[29]+sxyv3));
  kp[F(343)] += S*(+vv[8]-vv[25]-vv[32]-vv[37]+2*(txv[25]+tyv[25]+sxyv2));
  kp[F(344)] += S*(-vv[13]-vv[28]+vv[49]+vv[52]+tyv[13]+tyv[28]);
  kp[F(345)] += S*(+vv[12]-vv[29]-vv[48]+vv[53]+2*(tyv[29]-sxv3+sxyv3));
  kp[F(346)] += S*(+vv[45]+vv[57]-2*(txv[45]));
  kp[F(347)] += S*(+vv[1]-vv[4]+vv[16]+vv[21]-vv[41]+vv[61]-2*(txv[1]+txv[21]+txv[61]+sxyv1)-tyv[1]-tyv[16]-2*tyv[21]);
  kp[F(348)] += S*(+vv[9]+vv[24]-vv[33]-vv[36]-tyv[9]-tyv[24]);
  kp[F(349)] += S*(-vv[8]+vv[25]+vv[32]-vv[37]-2*(tyv[25]-sxv2+sxyv2));
  kp[F(350)] += S*(-vv[1]+vv[4]-vv[16]-vv[21]-vv[41]+vv[61]+2*(txv[1]+txv[21]+txv[41]+sxyv1)+tyv[1]+tyv[16]+2*tyv[21]);
  kp[F(351)] += S*(-vv[45]-vv[57]+2*txv[57]);
  kp[F(352)] += S*(-vv[12]-vv[29]-vv[48]+vv[53]+2*(txv[12]+sxv3));
  kp[F(353)] += S*(-vv[13]+vv[28]-vv[49]-vv[52]+2*txv[13]+tyv[13]-tyv[28]);
  kp[F(354)] += S*(+vv[44]-vv[56]-2*tyv[44]);
  kp[F(355)] += S*(+vv[0]+vv[5]+vv[17]-vv[20]-vv[40]-vv[60]+2*(-txv[5]+txv[20]+txv[40]+txv[60])-tyv[5]+tyv[20]+2*tyv[40]);
  kp[F(356)] += S*(+vv[13]-vv[28]-vv[49]-vv[52]+2*txv[28]-tyv[13]+tyv[28]);
  kp[F(357)] += S*(-vv[12]-vv[29]+vv[48]-vv[53]+2*(txv[29]+sxv3));
  kp[F(358)] += S*(+vv[45]-vv[57]-2*tyv[45]);
  kp[F(359)] += S*(-vv[1]+vv[4]+vv[16]+vv[21]-vv[41]-vv[61]+2*(txv[1]-txv[16]+txv[41]+txv[61])+tyv[1]-tyv[16]+2*tyv[41]);
  kp[F(360)] += S*(-vv[44]-vv[56]+2*(txv[44]+tyv[44]));
  kp[F(361)] += S*(-vv[45]-vv[57]+2*(txv[45]+tyv[45]));
  kp[F(362)] += S*(-vv[12]-vv[29]+vv[48]+vv[53]);
  kp[F(363)] += S*(+vv[8]+vv[25]+vv[32]+vv[37]-2*(txv[32]+tyv[32]+txv[37]+tyv[37]));
  kp[F(364)] += S*(+vv[0]-vv[5]+vv[17]+vv[20]+vv[40]-vv[60]-2*(txv[0]+txv[20]+txv[40]+sxv0)+tyv[5]-tyv[20]-2*tyv[40]);
  kp[F(365)] += S*(+vv[1]+vv[4]-vv[16]+vv[21]+vv[41]-vv[61]-2*(txv[1]+txv[21]+txv[41]+sxv1)-tyv[1]+tyv[16]-2*tyv[41]);
  kp[F(366)] += S*(+vv[8]+vv[25]-vv[32]-vv[37]+2*(tyv[32]+tyv[37]));
  kp[F(367)] += S*(+vv[12]+vv[29]+vv[48]+vv[53]-2*(txv[48]+txv[53]));
  kp[F(368)] += S*(+vv[8]+vv[25]+vv[32]-vv[37]-2*(txv[8]+sxv2));
  kp[F(369)] += S*(+vv[9]-vv[24]+vv[33]+vv[36]-2*txv[9]-tyv[9]+tyv[24]);
  kp[F(370)] += S*(-vv[0]-vv[5]-vv[17]+vv[20]+vv[40]+vv[60]-2*(-txv[5]+txv[20]+txv[40]+txv[60])+tyv[5]-tyv[20]-2*tyv[60]);
  kp[F(371)] += S*(+vv[44]-vv[56]+2*(tyv[56]));
  kp[F(372)] += S*(-vv[9]+vv[24]+vv[33]+vv[36]+tyv[9]-2*txv[24]-tyv[24]);
  kp[F(373)] += S*(+vv[8]+vv[25]-vv[32]+vv[37]-2*(txv[25]+sxv2));
  kp[F(374)] += S*(+vv[1]-vv[4]-vv[16]-vv[21]+vv[41]+vv[61]-2*(txv[1]-txv[16]+txv[41]+txv[61])-tyv[1]+tyv[16]-2*tyv[61]);
  kp[F(375)] += S*(+vv[45]-vv[57]+2*tyv[57]);
  kp[F(376)] += S*(-vv[0]+vv[5]-vv[17]-vv[20]+vv[40]-vv[60]
	       +2*(txv[0]+txv[20]+txv[60]+sxv0)-tyv[5]+tyv[20]+2*tyv[60]) ;
  kp[F(377)] += S*(-vv[1]-vv[4]+vv[16]-vv[21]+vv[41]-vv[61]
	       +2*(txv[1]+txv[21]+txv[61]+sxv1)+tyv[1]-tyv[16]+2*tyv[61]);
  kp[F(378)] += S*(-vv[8]-vv[25]-vv[32]-vv[37]+2*(txv[32]+txv[37]));
  kp[F(379)] += S*(-vv[12]-vv[29]+vv[48]+vv[53]-2*(tyv[48]+tyv[53]));
  kp[F(380)] += S*(+vv[44]+vv[56]-2*(txv[56]+tyv[56]));
  kp[F(381)] += S*(+vv[45]+vv[57]-2*(txv[57]+tyv[57]));
  kp[F(382)] += S*(-vv[12]-vv[29]-vv[48]-vv[53]+2*(txv[48]+tyv[48]+txv[53]+tyv[53]));
  kp[F(383)] += S*(+vv[8]+vv[25]-vv[32]-vv[37]);
  return ;
}

// returns 1 if something bad happened
__device__
static int
init_STV( const double xv[4] ,
	  const double yv[4] ,
	  struct QED_kernel_temps t ,
	  struct STV *k )
{
  const struct invariants Inv = set_invariants( xv , yv , t.Grid ) ;

  // Not needed
  // memset( k -> Sxv , 0 , 4*sizeof( double ) ) ;
  // memset( k -> Syv , 0 , 4*sizeof( double ) ) ;
  // memset( k -> Txv , 0 , 64*sizeof( double ) ) ;
  // memset( k -> Tyv , 0 , 64*sizeof( double ) ) ;
  // memset( k -> Vv  , 0 , 64*sizeof( double ) ) ;

  // precompuations for the interpolations
  const size_t ix1 = Inv.INVx.idx ;
  size_t ix2 = ix1+1 ;
  if( ix2 >= (size_t)t.Grid.nstpx ) {
    ix2 = ix1 ;
  }

  if( chnr_dS( xv, yv, Inv, t.Grid, k->Sxv, k->Syv ) ||
      chnr_dT( xv, yv, Inv, t.Grid, k->Txv, k->Tyv ) ||
      chnr_dV( xv, yv, Inv, t.Grid, k->Vv ) ) {
    return 1 ;
  }

  return 0 ;
}

// computes the kernel [rho - sigma];[mu][nu][lambda]
__device__
int
kernelQED( const double xv[4] ,
	   const double yv[4] ,
	   const struct QED_kernel_temps t ,
	   double kerv[6][4][4][4] )
{
  // FORNOW
  return 0;
  // FFs of QED kernel and their derivatives wrt x,cb,y
  struct STV x_y ;
  if( init_STV( xv, yv, t, &x_y ) ) return 1 ;

  // point out the kernel again
  double *kp = (double*)kerv ;
  CONSTRUCT_FULL_KERNEL<ident>( kp , 1.0, x_y ) ;

  return 0 ;
}

// computes the kernel [rho - sigma];[mu][nu][lambda]
__device__
int
kernelQED_axpy( const double xv[4] ,
                const double yv[4] ,
                const struct QED_kernel_temps t ,
                const struct STV x_y,
                double S,
                double kerv[6][4][4][4] )
{
  // FFs of QED kernel and their derivatives wrt x,cb,y
  // struct STV x_y;
  // if( init_STV( xv, yv, t, &x_y ) ) return 1 ;

  // point out the kernel again
  double *kp = (double*)kerv ;
  CONSTRUCT_FULL_KERNEL<ident>( kp , S, x_y ) ;
  // TODO: never need this?
  // CONSTRUCT_FULL_KERNEL<swap_mulam>( kp, -S, x_y ) ;

  return 0 ;
}
