/*
 BEGIN_JUCE_MODULE_DECLARATION

  ID:                 wray
  vendor:             lissajous
  version:            1.0.5
  name:               WRAY
  description:        Verified SIMD math for audio DSP
  website:            https://github.com/Lissajous-Labs/WRAY
  license:            GPL-2.0-or-later
  minimumCppStandard: 20

  dependencies:

 END_JUCE_MODULE_DECLARATION
*/

#pragma once

#include "simd/vec.h"
#include "denormals.h"
#include "numerics/polynomial.h"
#include "math.h"
#include "floating_point.h"
#include "adaa.h"
#include "filters/one_pole.h"
#include "filters/svf.h"
#include "filters/biquad.h"
