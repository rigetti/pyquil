##############################################################################
# Copyright 2017-2018 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################
# THIS FILE IS DERIVED AND MODIFIED FROM PROJECTQ. COPYRIGHT PROVIDED HERE:
#
#   Copyright 2017 ProjectQ-Framework (www.projectq.ch)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
##############################################################################


def get_default_settings():
    """
    Return the default settings for generating LaTeX circuits.

    settings is a dictionary with the following keys:
      gate_shadow: Whether or not to apply shadowing to the gates.
      lines: Settings for the lines.
      gates: Settings for the gates.
      control: Settings for the control symbols.

    :return: Default circuit settings
    :rtype: dict
    """
    settings = dict()
    settings['gate_shadow'] = True
    settings['lines'] = ({'style': 'very thin', 'double_classical': True,
                          'init_quantum': True, 'double_lines_sep': .04})
    settings['gates'] = ({'HGate': {'width': .5, 'offset': .3, 'pre_offset': .1},
                          'XGate': {'width': .35, 'height': .35, 'offset': .1},
                          'SwapGate': {'width': .35, 'height': .35, 'offset': .1},
                          'Rx': {'width': 1., 'height': .8, 'pre_offset': .2, 'offset': .3},
                          'Ry': {'width': 1., 'height': .8, 'pre_offset': .2, 'offset': .3},
                          'Rz': {'width': 1., 'height': .8, 'pre_offset': .2, 'offset': .3},
                          'EntangleGate': {'width': 1.8, 'offset': .2, 'pre_offset': .2},
                          'DeallocateQubitGate': {'height': .15, 'offset': .2, 'width': .2, 'pre_offset': .1},
                          'AllocateQubitGate': {'height': .15, 'width': .2, 'offset': .1, 'pre_offset': .1,
                                                'draw_id': False},
                          'MeasureGate': {'width': 0.75, 'offset': .2, 'height': .5, 'pre_offset': .2}})
    settings['control'] = {'size': .1, 'shadow': False}
    return settings


def header(settings):
    """
    Writes the Latex header using the settings file.

    The header includes all packages and defines all tikz styles.

    :param dictionary settings: LaTeX settings for document.
    :return: Header of the LaTeX document.
    :rtype: string
    """
    packages = (r"\documentclass{standalone}",
                r"\usepackage[margin=1in]{geometry}",
                r"\usepackage[hang,small,bf]{caption}",
                r"\usepackage{tikz}",
                r"\usepackage{braket}",
                r"\usetikzlibrary{backgrounds,shadows.blur,fit,decorations.pathreplacing,shapes}")

    init = (r"\begin{document}",
            r"\begin{tikzpicture}[scale=0.8, transform shape]")

    gate_style = (r"\tikzstyle{basicshadow}="
                  r"[blur shadow={shadow blur steps=8, shadow xshift=0.7pt, shadow yshift=-0.7pt, shadow scale=1.02}]")

    if not (settings.get('gate_shadow') or settings.get('control', {}).get('shadow')):
        gate_style = ""

    gate_style += r"\tikzstyle{basic}=[draw,fill=white,"
    if settings.get('gate_shadow'):
        gate_style += "basicshadow"
    gate_style += "]\n"

    gate_style += ("\\tikzstyle{{operator}}=[basic,minimum size=1.5em]\n\\tikzstyle{{phase}}=[fill=black,shape=circle,"
                   "minimum size={}"
                   "cm,inner sep=0pt,outer sep=0pt,draw=black").format(settings.get('control', {}).get('size', 0))
    if settings.get('control', {}).get('shadow'):
        gate_style += ",basicshadow"
    gate_style += ("]\n\\tikzstyle{none}=[inner sep=0pt,outer sep=-.5pt,"
                   "minimum height=0.5cm+1pt]\n"
                   "\\tikzstyle{measure}=[operator,inner sep=0pt,minimum "
                   + "height={}cm, minimum width={}cm]\n".format(
                       settings.get('gates', {}).get('MeasureGate', {}).get('height', 0),
                       settings.get('gates', {}).get('MeasureGate', {}).get('width', 0))
                   + "\\tikzstyle{xstyle}=[circle,basic,minimum height=")
    x_gate_radius = min(settings.get('gates', {}).get('XGate', {}).get('height', 0),
                        settings.get('gates', {}).get('XGate', {}).get('width', 0))
    gate_style += ("{x_rad}cm,minimum width={x_rad}cm,inner sep=0pt,{linestyle}]\n").format(
        x_rad=x_gate_radius,
        linestyle=settings.get('lines', {}).get('style', ""))
    if settings.get('gate_shadow'):
        gate_style += ("\\tikzset{\nshadowed/.style={preaction={transform canvas={shift={(0.5pt,-0.5pt)}},"
                       " draw=gray, opacity=0.4}},\n}\n")
    gate_style += "\\tikzstyle{swapstyle}=[inner sep=-1pt, outer sep=-1pt, minimum width=0pt]"
    edge_style = ("\\tikzstyle{edgestyle}=[" + settings.get('lines', {}).get('style', "") + "]")
    return "\n".join(("\n".join(packages), "\n".join(init), gate_style, edge_style))


def footer():
    """
    Return the footer of the LaTeX document.

    :return: LaTeX document footer.
    :rtype: string
    """
    return "\\end{tikzpicture}\n\\end{document}"
