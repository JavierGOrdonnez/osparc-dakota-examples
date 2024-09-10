# import s4l_v1 as s4l
# from s4l_v1._api.application import get_app_safe, run_application
# import XCoreModeling as xcm
# import XCore
# import s4l_neurofunctions as nf
import numpy as np
import os, sys
import pathlib as pl
import datetime

### VERY CRIPPLED VERSION OF THE FUNCTION
## Just have same inputs, assertions, and outputs


# base_path = os.getcwd()  ## for cloud-based execution -- workspace
# print("base path: ", base_path)
# sys.path.append(base_path)
# from OptiStim.Modeling import (
#     create_layered_model,
#     import_nerve_model,
# )
# from OptiStim.EMSimulations import setup_emsim

################################################
################################################
WIDTH_X = 80
WIDTH_Y = 80
TOTAL_THICKNESS = 12.5
NERVE_LENGTH = 30
ANODE = "Electrode_1"
CATHODE = "Electrode_3"
GRID_SIZE = 0.5
DL = 0.2
# NEURON_DIAMETER = 4.0  # um
# NEURON_DIAMETERS = np.arange(10, dtype=float) + 1.0  # 1.0 to 10.0 um
NEURON_DIAMETERS = np.arange(4, 10 + 1, dtype=float)  # 4.0 to 10.0 um
# 4um is min allowed diameter for Sensory MRG -- should we change to Small MRG?

"""
def main():
    print(list(os.environ.keys()))
    input_path_json = pl.Path(os.environ["INPUT_FOLDER"])
    input_path_models = pl.Path(os.environ["INPUT_FOLDER"]) / "Inputs"
    output_path = pl.Path(os.environ["OUTPUT_FOLDER"])

    input_file_path = input_path_json / "input.json"
    output_file_path = output_path / "output.json"

    input = json.loads(input_file_path.read_text())

    output = evaluate(**input)

    output_file_path.write_text(json.dumps(output))
"""

## Manually added; dont want it to be an input.
INPUTS_PATH = pl.Path(os.getcwd()) / "Inputs"


def evaluate(
    RELDEPTH=0.5,  ## NEW: relative depth within the SCT
    DIAMETER=0.3,
    POSITION=-14.0,
    ANGLE=90,
    ELECTRELDEPTH=0.5,  ## relative depth of the electrode within the LAT
    #
    THICKNESS_SKIN=0.5,  ## epidermis + dermis
    THICKNESS_SCT=0.95,  ## subcutaneous tissue
    THICKNESS_APONEUROSIS=0.4,  ## aponeurosis / galea
    THICKNESS_LOOSE_AREOLAR_TISSUE=0.65,
    THICKNESS_SKULL_OUTER=1.83,
    THICKNESS_SKULL_DIPLOE=2.84,
    THICKNESS_SKULL_INNER=1.44,
    THICKNESS_CSF=3.6,  ## https://www.researchgate.net/figure/Thickness-of-scalp-skull-and-CSF-layer_fig14_51754044
    #
    CONDUCTIVITY_SKIN=None,
    CONDUCTIVITY_SCT=None,
    CONDUCTIVITY_APONEUROSIS=None,
    CONDUCTIVITY_LOOSE_AREOLAR_TISSUE=None,
    CONDUCTIVITY_SKULL_CORTICAL=None,
    CONDUCTIVITY_SKULL_DIPLOE=None,
    CONDUCTIVITY_CSF=None,
) -> float:
    """This evaluation function creates a microscale model (layered) and creates a
    rectilinear grid to evaluate the electric field (its derivative, the AF) at the center of the nerve structure.

    Args:
        input_path_models (Path): Path to the input folder with the models (Salvia electrode and nerve histology).
        DEPTH (float): Depth of the nerve below the skin surface. It must be within the SubcutaneousTissue
                - therefore, between 2.8 and 5.2 mm, minus the radius of the nerve itself, and a bit more (0.2mm) to avoid
                merge issues. Thus, the range is [3.0 + D/2, 5.0 - D/2] with D the diameter of the nerve.
        DIAMETER (float): Diameter of the nerve structure. The range is 0.1 to 2.0 mm.
        ANGLE (float): Angle of the nerve with respect to the implant (which will be rotated).
                Angle 0 is parallel, angle 90 is perpendicular. The range is 0-180.
        POSITION (float): Horizontal position of the nerve with respect to the implant. Position zero is the exact center
                above the implant. The range is -25mm to +25mm.

    Returns:
        AFmax: Peak of the Activating Function along the center of the nerve structure. This quantity (in V/s) is a fairly
                good indicator of the activation of neural fibers due to an electric field.
    """
    print("Evaluating model...")
    DEPTH = np.round(DEPTH, 3)
    DIAMETER = np.round(DIAMETER, 3)
    ANGLE = np.round(ANGLE, 2)
    POSITION = np.round(POSITION, 2)

    # if not (
    #     (DEPTH - DIAMETER / 2 >= THICKNESS_SKIN)
    #     and (DEPTH + DIAMETER / 2 <= THICKNESS_SKIN + THICKNESS_SCT)
    # ):
    #     print(
    #         f"Nerve must be within the subcutaneous tissue layer. \n"
    #         f"\tCurrent upper nerve depth: {DEPTH - DIAMETER / 2}\n"
    #         f"\tSCT beginning: {THICKNESS_SKIN}\n"
    #         f"\tCurrent lower nerve depth: {DEPTH + DIAMETER / 2}\n"
    #         f"\tSCT end: {THICKNESS_SKIN + THICKNESS_SCT}\n"
    #     )
    #     return np.nan

    current_time = datetime.datetime.now().strftime("%Y%m%d.%H%M%S%d")
    model_random_index = np.random.randint(1000)
    model_name = f"MicroScale_Rectilinear_{current_time}_{model_random_index}"  # avoid identical names
    model_path = os.path.join(INPUTS_PATH, f"{model_name}.smash")
    print("Trying to open S4L application...")
    # if get_app_safe() is None:
    #     run_application(
    #         disable_ui_plugins=True,
    #     )
    # print("S4L application opened!")
    # app = get_app_safe()

    # ## JUST FOR TESTING
    # afdo = nf.af.AFDataObject(af_type="AF")
    # afdo.load(INPUTS_PATH)
    # #
    # gafc = nf.af.GAFCalculatorHeterogeneous(
    #     dst=10, kernels_path=INPUTS_PATH / "GAF_MRG_kernels"
    # )
    # pulse = nf.neuron.StimulationPulse(INPUTS_PATH / "01ms_burst_stimulation.txt")
    # gafc.compute_gaf(afdo, pulse, MODE="BruteForce")
    # gafc.save_peaks(INPUTS_PATH.parent)

    # gafmax = gafc.get_peaks().get_gaf_data().AF_max

    # return gafmax.values[0]

    print("Creating new S4L document...")
    # s4l.document.New()
    # s4l.document.SaveAs(model_path)

    # implant = create_layered_model.import_implant(
    #     INPUTS_PATH,
    #     "Implant_Salvia_E123_fixedsupport.sab",
    # )
    # create_layered_model.main(
    #     WIDTH_X=WIDTH_X,
    #     WIDTH_Y=WIDTH_Y,
    #     model_name=model_name,
    #     include_pericraneum=False,
    #     include_gm=False,
    #     include_wm=False,
    #     TOTAL_THICKNESS=TOTAL_THICKNESS,
    #     THICKNESS_SKIN=THICKNESS_SKIN,
    #     THICKNESS_SCT=THICKNESS_SCT,
    #     THICKNESS_APONEUROSIS=THICKNESS_APONEUROSIS,
    #     THICKNESS_LOOSE_AREOLAR_TISSUE=THICKNESS_LOOSE_AREOLAR_TISSUE,
    #     THICKNESS_SKULL_OUTER=THICKNESS_SKULL_OUTER,
    #     THICKNESS_SKULL_DIPLOE=THICKNESS_SKULL_DIPLOE,
    #     THICKNESS_SKULL_INNER=THICKNESS_SKULL_INNER,
    #     THICKNESS_CSF=THICKNESS_CSF,
    # )

    ## RELATIVE DEPTH OF ELECTRODE
    # the electrode, to start with, is already on top of the skull
    ## Then can move up using the relative depth
    # lead = nf.utils.model.get_entity("Salvia_Lead")
    LEAD_THICKNESS = 0.2
    TOTAL_SKIN_THICKNESS = (
        THICKNESS_SKIN
        + THICKNESS_SCT
        + THICKNESS_APONEUROSIS
        + THICKNESS_LOOSE_AREOLAR_TISSUE
    )
    if LEAD_THICKNESS >= THICKNESS_LOOSE_AREOLAR_TISSUE:
        raise ValueError(
            "Lead thickness is too large for the loose areolar tissue layer. Not moving the lead."
        )

    else:
        lead_depth = TOTAL_SKIN_THICKNESS
        lead_depth -= LEAD_THICKNESS  ## THIS IS WRONG IN MODEL CREATION (SHOULD BE 1/2). Anyway lets fix it here.
        ## this is current depth. Need to move to new depth

        LOWER_BOUND = (
            TOTAL_SKIN_THICKNESS - THICKNESS_LOOSE_AREOLAR_TISSUE + 0.5 * LEAD_THICKNESS
        )

        UPPER_BOUND = TOTAL_SKIN_THICKNESS - 0.5 * LEAD_THICKNESS
        LEAD_DEPTH = LOWER_BOUND + ELECTRELDEPTH * (UPPER_BOUND - LOWER_BOUND)
        assert LEAD_DEPTH + 0.5 * LEAD_THICKNESS <= TOTAL_SKIN_THICKNESS
        ## sanity check, remove for OSPARC
        # lead.ApplyTransform(
        #     s4l.model.Translation(xcm.Vec3(0, 0, LEAD_DEPTH - lead_depth))
        # )

    # umesh_nerve = import_nerve_model.main(
    #     "L6_2D_Mesh_iSEG.sab",
    #     INPUTS_PATH,
    #     L=NERVE_LENGTH,
    #     DEPTH=DEPTH,
    #     dl=DL,
    # )
    # nerve_group = xcm.EntityGroup()
    # nerve_group.Name = "Nerve_" + model_name
    # nerve_group.Add(xcm.ExtractUnstructuredMeshSurface(umesh_nerve["InnerNerve"]))
    # nerve_group.Add(xcm.ExtractUnstructuredMeshSurface(umesh_nerve["OuterNerve"]))
    # umesh_nerve.Delete()
    # del umesh_nerve

    # ############################################################
    # ## apply the parameters, before the merge
    # ### Move nerve to center, to apply scaling and rotation around the center
    # v1, v2 = xcm.GetBoundingBox(nerve_group.Entities)
    # center_pos = nf.utils.model.get_center_position(nerve_group)
    # nerve_group.ApplyTransform(s4l.model.Translation(-center_pos))

    # ## DIAMETER
    # original_diameter = np.abs((v1 - v2)[0])
    # if DIAMETER is None:
    #     DIAMETER = original_diameter
    # else:
    #     nerve_group.ApplyTransform(
    #         s4l.model.Scaling(
    #             xcm.Vec3(DIAMETER / original_diameter, 1, DIAMETER / original_diameter)
    #         )
    #     )

    # ## ANGLE
    # implant = nf.get_entity("Salvia_Lead")
    # implant.ApplyTransform(
    #     s4l.model.Rotation(xcm.Vec3(0, 0, 1), (90 - ANGLE) * np.pi / 180)
    # )

    ## DEPTH
    if THICKNESS_SCT < DIAMETER:
        DEPTH = THICKNESS_SKIN + 0.5 * THICKNESS_SCT
        ## RELDEPTH is meaningless in this case - just place nerve at center of SCT
    else:
        LOWER_BOUND = THICKNESS_SKIN + 0.5 * DIAMETER
        UPPER_BOUND = THICKNESS_SKIN + THICKNESS_SCT - 0.5 * DIAMETER
        DEPTH = LOWER_BOUND + RELDEPTH * (UPPER_BOUND - LOWER_BOUND)
    # nerve_group.ApplyTransform(s4l.model.Translation(xcm.Vec3(0, 0, DEPTH)))
    # center_spline = import_nerve_model.create_spline(NERVE_LENGTH * 0.9, DEPTH)

    # ## POSITION (and ANGLE)
    # nerve_group.ApplyTransform(
    #     s4l.model.Translation(
    #         xcm.Vec3(
    #             POSITION * np.cos(np.pi * (90 - ANGLE) / 180),
    #             POSITION * np.sin(np.pi * (90 - ANGLE) / 180),
    #             0,
    #         )
    #     )
    # )
    # center_spline.ApplyTransform(
    #     s4l.model.Translation(
    #         xcm.Vec3(
    #             POSITION * np.cos(np.pi * (90 - ANGLE) / 180),
    #             POSITION * np.sin(np.pi * (90 - ANGLE) / 180),
    #             0,
    #         )
    #     )
    # )

    # # axon = nf.neuron.model.create_neuron_from_spline(
    # #     center_spline, "Sensory MRG", diameter=NEURON_DIAMETER
    # # )
    # axons = [
    #     nf.neuron.model.create_neuron_from_spline(
    #         center_spline, "Sensory MRG", diameter=d
    #     )
    #     for d in NEURON_DIAMETERS
    # ]

    # XCore.ProcessPendingMainAppSignals()  # MGui - avoid ACIS error. Call bfr sim setup.
    # emsim_nerve = setup_emsim.main(
    #     model_name,
    #     anode=ANODE,
    #     cathode=CATHODE,
    #     grid_size=GRID_SIZE,
    #     nerve_name="Nerve_" + model_name,
    #     nerve_grid_size=np.round(0.02 * DIAMETER, 3),
    #     CONDUCTIVITY_SKIN=CONDUCTIVITY_SKIN,
    #     CONDUCTIVITY_SCT=CONDUCTIVITY_SCT,
    #     CONDUCTIVITY_APONEUROSIS=CONDUCTIVITY_APONEUROSIS,
    #     CONDUCTIVITY_LOOSE_AREOLAR_TISSUE=CONDUCTIVITY_LOOSE_AREOLAR_TISSUE,
    #     CONDUCTIVITY_SKULL_CORTICAL=CONDUCTIVITY_SKULL_CORTICAL,
    #     CONDUCTIVITY_SKULL_DIPLOE=CONDUCTIVITY_SKULL_DIPLOE,
    #     CONDUCTIVITY_CSF=CONDUCTIVITY_CSF,
    # )
    # print("Ready to run simulation...")
    # nf.save_and_close_model()
    return {"AFmax_4um": 0.0, "GAFmax_4um": 0.0}  # mockup, for now
    emsim_nerve.RunSimulation(wait=True)
    # nf.utils.simulation.RunSimulationWithSubprocess(emsim_nerve, wait=True)
    print("Simulation done!")

    # AFmax = get_AF_max(emsim_nerve, axon)
    # print(f"AFmax {model_name}: {AFmax}")
    mafe = nf.af.MultiAFExtractor(axons)
    mafe.extract_af(emsim_nerve)
    results = {
        f"AFmax_{int(d)}um": np.nanmax(mafe.get_af(axon.Name, emsim_nerve.Name))
        for d, axon in zip(NEURON_DIAMETERS, axons)
    }

    gafc = nf.af.GAFCalculatorHeterogeneous(
        dst=10, kernels_path=INPUTS_PATH / "GAF_MRG_kernels"
    )
    pulse = nf.neuron.StimulationPulse(INPUTS_PATH / "01ms_burst_stimulation.txt")
    gafc.compute_gaf(mafe.af_data_object, pulse, MODE="BruteForce")
    results.update(
        {
            f"GAFmax_{int(d)}um": gafc.get_peaks()
            .get_gaf_data(axon.Name, emsim_nerve.Name, pulse.name)
            .AF_max.values[0]
            for d, axon in zip(NEURON_DIAMETERS, axons)
        }
    )

    print("Cleaning up...")
    nf.utils.simulation.remove_all_simulations()
    s4l.document.Save()  # save the model (enable later inspection)
    s4l.document.New()  # close the model
    # os.remove(model_path)  # remove the model file - otherwise will blow disk space

    ## NB: Dakota is minimization, so we need to return the negative of the AFmax
    ## NB: not doing optimization anymore!
    # return AFmax

    # return results["AFmax_4um"]  ## FIXME for now, keep returning just one value
    return results


if __name__ == "__main__":
    # main()

    # JGO: for local testing
    # evaluate()

    ### Weird values in training set 800LHS 18D
    # 73
    # 1.323701782    0.1701655891   46.40729694    0.7232063245   0.6105168418   0.1731109139   0.09540467221  1.390186105    3.074015809    1.981461753    3.720179775    0.2005719032   0.02278952185  0.05378767575  0.05207343783  0.00650233921  0.1325314686   2.773602514
    # 5901.033467
    # evaluate(
    #     DEPTH=1.323701782,
    #     DIAMETER=0.1701655891,
    #     ANGLE=46.40729694,
    #     THICKNESS_SKIN=0.7232063245,
    #     THICKNESS_SCT=0.6105168418,
    #     THICKNESS_APONEUROSIS=0.1731109139,
    #     THICKNESS_LOOSE_AREOLAR_TISSUE=0.09540467221,
    #     THICKNESS_SKULL_OUTER=1.390186105,
    #     THICKNESS_SKULL_DIPLOE=3.074015809,
    #     THICKNESS_SKULL_INNER=1.981461753,
    #     THICKNESS_CSF=3.720179775,
    #     CONDUCTIVITY_SKIN=0.2005719032,
    #     CONDUCTIVITY_SCT=0.02278952185,
    #     CONDUCTIVITY_APONEUROSIS=0.05378767575,
    #     CONDUCTIVITY_LOOSE_AREOLAR_TISSUE=0.05207343783,
    #     CONDUCTIVITY_SKULL_CORTICAL=0.00650233921,
    #     CONDUCTIVITY_SKULL_DIPLOE=0.1325314686,
    #     CONDUCTIVITY_CSF=2.773602514,
    # )

    # 434
    # 1.301338378    0.1795140531   12.04429255    0.3228122466   0.4089584928   0.1714257138   0.7880841609   1.913812388    2.569829736    2.061908981    3.150595556    0.2178367043   0.1020386211   0.08431107606  0.09095780158  0.01036268517  0.2698423494   1.737080873
    # 3605.685524

    # 406
    # 1.320541809    0.2185884819   16.83843397    0.6817609182   0.6529094339   0.291380371    0.07781948262  1.42073382     1.944157254    1.889597955    2.611902003    0.1305397344   0.1300138572   0.1139971393   0.06089555367  0.001494577037 0.081041841    1.674414428
    # 2442.158419

    # 567
    # 1.279063597    0.1570555491   29.18163381    0.7226917511   0.5412788987   0.09610805631  0.3574279293   1.021568434    2.680733762    1.861905027    3.898353005    0.231312198    0.1471170559   0.0727588391   0.06737688401  0.004326733506 0.2571080826   3.052525092
    # 2875.512662

    ### after filtering (mild nerve-sct rule)
    #      %eval_id     DEPTH  DIAMETER      ANGLE  THICKNESS_SKIN  THICKNESS_SCT  \
    # 250     251.0  1.027943  0.133112  82.761115        0.487321       0.928833
    # 207     208.0  0.991211  0.199553   4.353060        0.444434       0.644857
    # 615     616.0  0.622396  0.145410  22.720081        0.530278       0.433037
    # 591     592.0  1.251824  0.202909  73.114421        0.363408       0.916653
    # 130     131.0  0.848245  0.359155   5.482023        0.381732       0.756452

    #      THICKNESS_APONEUROSIS  THICKNESS_LOOSE_AREOLAR_TISSUE  \
    # 250               0.122783                        0.315074
    # 207               0.430989                        0.300956
    # 615               0.053066                        0.356939
    # 591               0.380346                        0.310260
    # 130               0.052120                        0.304249

    #      THICKNESS_SKULL_OUTER  THICKNESS_SKULL_DIPLOE  THICKNESS_SKULL_INNER  \
    # 250               1.587005                2.530165               1.993957
    # 207               0.884333                1.882952               1.704233
    # 615               1.978513                1.870060               2.138922
    # 591               1.980010                2.059712               1.439264
    # 130               2.784831                3.011241               1.738492

    #      THICKNESS_CSF  CONDUCTIVITY_SKIN  CONDUCTIVITY_SCT  \
    # 250       3.561414           0.229314          0.050474
    # 207       3.001046           0.213196          0.086685
    # 615       4.018483           0.088913          0.241981
    # 591       2.994648           0.100166          0.256761
    # 130       3.019233           0.165324          0.144064

    #      CONDUCTIVITY_APONEUROSIS  CONDUCTIVITY_LOOSE_AREOLAR_TISSUE  \
    # 250                  0.063549                           0.057989
    # 207                  0.111603                           0.099427
    # 615                  0.086154                           0.050900
    # 591                  0.115036                           0.102837
    # 130                  0.088057                           0.062486

    #      CONDUCTIVITY_SKULL_CORTICAL  CONDUCTIVITY_SKULL_DIPLOE  CONDUCTIVITY_CSF  \
    # 250                     0.009967                   0.076047          1.825783
    # 207                     0.006776                   0.091433          2.509497
    # 615                     0.009685                   0.054291          3.151137
    # 591                     0.011495                   0.087128          2.303623
    # 130                     0.006707                   0.083435          0.966493

    #          -AFpeak
    # 250  1320.059896
    # 207  1090.192403
    # 615  1038.547750
    # 591   958.958791
    # 130   902.247249

    evaluate(
        DEPTH=1.027943,
        DIAMETER=0.133112,
        ANGLE=82.761115,
        THICKNESS_SKIN=0.487321,
        THICKNESS_SCT=0.928833,
        THICKNESS_APONEUROSIS=0.122783,
        THICKNESS_LOOSE_AREOLAR_TISSUE=0.315074,
        THICKNESS_SKULL_OUTER=1.587005,
        THICKNESS_SKULL_DIPLOE=2.530165,
        THICKNESS_SKULL_INNER=1.993957,
        THICKNESS_CSF=3.561414,
        CONDUCTIVITY_SKIN=0.229314,
        CONDUCTIVITY_SCT=0.050474,
        CONDUCTIVITY_APONEUROSIS=0.063549,
        CONDUCTIVITY_LOOSE_AREOLAR_TISSUE=0.057989,
        CONDUCTIVITY_SKULL_CORTICAL=0.009967,
        CONDUCTIVITY_SKULL_DIPLOE=0.076047,
        CONDUCTIVITY_CSF=1.825783,
    )
